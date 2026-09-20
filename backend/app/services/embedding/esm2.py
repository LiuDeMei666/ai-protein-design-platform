"""ESM-2 蛋白语言模型服务：序列嵌入 + 掩码边缘打分。

零样本突变打分的核心
--------------------
对突变位点做 ``<mask>`` 掩码后前向，取该位点的对数概率分布，定义

    ΔlogP(mut) = logP(mut | context) - logP(wt | context)

该值越大，说明语言模型认为替换越"自然"，是稳定性/功能影响的零样本代理，
无需任何标注数据。两条通道：

* :meth:`ESM2Service.masked_marginal` —— **准确通道**。逐位点掩码，批量前向。
* :meth:`ESM2Service.wild_type_logprobs` —— **快速通道**。单次前向（不掩码），
  用于超长序列的位点预筛，SQL 级别的成本换全位点覆盖。

工程要点
--------
* HF 缓存重定向到 ``<项目>/models/hf_cache``，不污染本机既有 165GB 缓存。
* 权重加载三级降级：主模型 GPU -> 小模型 GPU -> 小模型 CPU，任何一级失败自动下落。
* 超长序列（>1022 残基）走滑窗 + 重叠区平均，边界信息不丢失，且在结果中标注窗口。
* 嵌入结果按 ``sha256(序列) + 模型`` 落盘缓存，重复分析零成本。
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...core.config import get_settings
from ...core.errors import ModelNotAvailableError
from ...core.logging import describe_sequence, get_logger

logger = get_logger(__name__)

#: 与蛋白语言模型 20 字母表一致的固定顺序（所有下游打分矩阵列顺序以此为准）
AA_ORDER: str = "ACDEFGHIKLMNPQRSTVWY"
#: ESM-2 位置上限 1024，预留 <cls>/<eos>
CONTEXT_LIMIT: int = 1022


# --------------------------------------------------------------------------- #
# HF 环境准备（必须在 import transformers 之前完成）
# --------------------------------------------------------------------------- #
def prepare_hf_env() -> Path:
    """设置 HF_HOME / HF_ENDPOINT，返回缓存根目录。幂等。"""
    settings = get_settings()
    hf_home = Path(settings.models_dir) / "hf_cache"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return hf_home


def model_cache_dir(repo_id: str, hf_home: Path | None = None) -> Path:
    """返回某模型在本地 HF 缓存中的目录。"""
    root = hf_home or (Path(get_settings().models_dir) / "hf_cache")
    return root / "hub" / ("models--" + repo_id.replace("/", "--"))


def is_model_cached(repo_id: str, min_mb: float = 100.0) -> bool:
    """判断模型权重是否已完整落盘（不触发下载）。"""
    directory = model_cache_dir(repo_id)
    if not directory.exists():
        return False
    total = 0
    for item in directory.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total > min_mb * 1024 * 1024


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class EmbeddingResult:
    """序列嵌入结果。"""

    per_residue: np.ndarray  # [L, D] float32
    mean: np.ndarray  # [D] float32
    layer: int
    windows: list[tuple[int, int]]
    from_cache: bool = False

    @property
    def length(self) -> int:
        return int(self.per_residue.shape[0])

    @property
    def dim(self) -> int:
        return int(self.per_residue.shape[1])


@dataclass
class MaskedMarginalResult:
    """逐位点掩码边缘打分结果。"""

    positions: list[int]
    aa_order: str
    #: [n_positions, 20]，值为 ΔlogP（突变型 - 野生型），正数表示模型偏好该替换
    delta_logprob: np.ndarray
    #: [n_positions] 野生型残基在该上下文下的对数概率
    wt_logprob: np.ndarray
    windows: list[tuple[int, int]]

    def score(self, position: int, mutant_aa: str) -> float:
        """查询某个具体突变的 ΔlogP；缺失时返回 0.0。"""
        if position not in self.positions:
            return 0.0
        row = self.positions.index(position)
        if mutant_aa not in self.aa_order:
            return 0.0
        col = self.aa_order.index(mutant_aa)
        return float(self.delta_logprob[row, col])


# --------------------------------------------------------------------------- #
# 服务实现
# --------------------------------------------------------------------------- #
class ESM2Service:
    """ESM-2 推理服务（进程内单例，惰性加载）。"""

    def __init__(self) -> None:
        self._load_lock = threading.RLock()
        self._infer_lock = threading.RLock()
        self._loaded = False
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_name: str = ""
        self._device: str = "cpu"
        self._use_fp16 = False
        self._degradation_reason: str | None = None
        self._aa_ids: list[int] = []
        self._mask_id: int = 0
        self._hidden_size: int = 0

    # ---------------- 状态查询（不触发加载） ---------------- #
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    def status(self) -> dict[str, Any]:
        """供健康检查使用的状态快照（不加载模型）。"""
        settings = get_settings()
        hf_home = prepare_hf_env()

        resolved_device = "cpu"
        if settings.device == "cuda":
            resolved_device = "cuda"
        elif settings.device == "auto":
            try:
                import torch

                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                resolved_device = "cpu"

        return {
            "model": settings.esm_model,
            "fallback_model": settings.esm_fallback_model,
            "resolved_device": resolved_device,
            "requested_device": settings.device,
            "use_fp16": settings.use_fp16 and resolved_device == "cuda",
            "hf_endpoint": settings.hf_endpoint,
            "hf_home": str(hf_home),
            "main_model_cached": is_model_cached(settings.esm_model),
            "fallback_model_cached": is_model_cached(settings.esm_fallback_model),
            "loaded": self._loaded,
            "loaded_model": self._model_name or None,
            "loaded_device": self._device if self._loaded else None,
            "degradation_reason": self._degradation_reason,
        }

    # ---------------- 模型加载 ---------------- #
    def _resolve_device(self) -> str:
        settings = get_settings()
        if settings.device == "cpu":
            return "cpu"
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ModelNotAvailableError(f"未安装 torch: {exc}") from exc
        if settings.device == "cuda" and not torch.cuda.is_available():
            logger.warning("配置要求 cuda 但不可用，回退 CPU")
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _try_load(self, model_name: str, device: str, use_fp16: bool) -> None:
        """尝试加载指定模型/设备组合，失败抛异常由上层降级。"""
        prepare_hf_env()
        from transformers import AutoTokenizer, EsmForMaskedLM  # 延迟导入

        logger.info("加载 ESM-2: %s (device=%s, fp16=%s)", model_name, device, use_fp16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmForMaskedLM.from_pretrained(model_name)
        model.eval()

        if device == "cuda":
            model = model.to("cuda")
            if use_fp16:
                model = model.half()

        self._tokenizer = tokenizer
        self._model = model
        self._model_name = model_name
        self._device = device
        self._use_fp16 = use_fp16 and device == "cuda"
        self._aa_ids = [tokenizer.convert_tokens_to_ids(aa) for aa in AA_ORDER]
        self._mask_id = tokenizer.mask_token_id
        self._hidden_size = int(getattr(model.config, "hidden_size", 0))
        self._loaded = True

    def ensure_loaded(self) -> None:
        """确保模型就绪；三级降级：主模型 GPU -> 小模型 GPU -> 小模型 CPU。"""
        if self._loaded:
            return

        with self._load_lock:
            if self._loaded:
                return

            settings = get_settings()
            device = self._resolve_device()
            wants_fp16 = settings.use_fp16 and device == "cuda"

            candidates: list[tuple[str, str, bool, str]] = [
                (settings.esm_model, device, wants_fp16, ""),
                (
                    settings.esm_fallback_model,
                    device,
                    wants_fp16,
                    f"主模型 {settings.esm_model} 加载失败",
                ),
                (
                    settings.esm_fallback_model,
                    "cpu",
                    False,
                    f"GPU 推理不可用",
                ),
            ]

            errors: list[str] = []
            for model_name, target_device, fp16, reason in candidates:
                try:
                    self._try_load(model_name, target_device, fp16)
                    if reason:
                        self._degradation_reason = f"{reason}，已降级为 {model_name} @ {target_device}"
                        logger.warning("模型降级: %s", self._degradation_reason)
                    logger.info(
                        "ESM-2 就绪: %s @ %s (hidden=%d)", model_name, target_device, self._hidden_size
                    )
                    return
                except Exception as exc:
                    errors.append(f"{model_name}@{target_device}: {exc}")
                    logger.warning("加载 %s@%s 失败: %s", model_name, target_device, exc)

            raise ModelNotAvailableError(
                "ESM-2 权重加载失败，请先执行 python scripts/download_esm2.py",
                detail={
                    "attempts": errors,
                    "hint": "HF 直连不可用，必须走镜像 HF_ENDPOINT=https://hf-mirror.com",
                },
            )

    # ---------------- 缓存 ---------------- #
    def _cache_path(self, sequence: str, kind: str) -> Path:
        settings = get_settings()
        slug = self._model_name.replace("/", "--") or "unloaded"
        digest = hashlib.sha256(
            f"{kind}|{sequence}|{slug}|{settings.max_embedding_length}".encode()
        ).hexdigest()
        directory = Path(settings.cache_dir) / "embeddings" / slug
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{digest}.npz"

    # ---------------- 窗口规划 ---------------- #
    @staticmethod
    def plan_windows(length: int, limit: int = CONTEXT_LIMIT) -> list[tuple[int, int]]:
        """把序列切成不超过 ``limit`` 的窗口，50% 重叠以便重叠区平均。"""
        if length <= limit:
            return [(0, length)]
        stride = max(1, limit // 2)
        windows: list[tuple[int, int]] = []
        start = 0
        while start < length:
            end = min(start + limit, length)
            windows.append((start, end))
            if end >= length:
                break
            start += stride
        return windows

    @staticmethod
    def _position_windows(length: int, positions: list[int]) -> dict[int, list[int]]:
        """把位点分配到各自所属的上下文窗口（窗口以位点为中心，便于掩码）。"""
        limit = CONTEXT_LIMIT
        if length <= limit:
            return {0: list(positions)}

        half = limit // 2
        groups: dict[int, list[int]] = {}
        for position in positions:
            start = max(0, min(position - half, length - limit))
            groups.setdefault(start, []).append(position)
        return groups

    # ---------------- 前向 ---------------- #
    def _forward(self, input_ids, output_hidden_states: bool = False):
        """统一前向入口（含 autocast）。

        Args:
            input_ids: token 张量。
            output_hidden_states: 取嵌入时必须置 ``True``，否则
                ``outputs.hidden_states`` 为 ``None``。
        """
        import torch

        kwargs = {"input_ids": input_ids}
        if output_hidden_states:
            kwargs["output_hidden_states"] = True

        with self._infer_lock:
            with torch.no_grad():
                if self._use_fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self._model(**kwargs)
                else:
                    outputs = self._model(**kwargs)
        return outputs

    def _encode(self, sequence: str):
        """tokenize 并校验长度不超过模型位置上限。"""
        encoded = self._tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        token_count = int(encoded["input_ids"].shape[1])
        if token_count > 1024:
            raise ModelNotAvailableError(
                f"窗口长度 {token_count} 超出 ESM-2 位置上限 1024",
                detail={"hint": "内部窗口规划错误"},
            )
        return encoded["input_ids"]

    # ---------------- 嵌入 ---------------- #
    def embeddings(self, sequence: str, use_cache: bool = True) -> EmbeddingResult:
        """计算逐残基嵌入与均值嵌入。"""
        self.ensure_loaded()

        import torch

        cache_file = self._cache_path(sequence, kind="embedding")
        if use_cache and cache_file.exists():
            try:
                payload = np.load(cache_file)
                return EmbeddingResult(
                    per_residue=payload["per_residue"],
                    mean=payload["mean"],
                    layer=int(payload["layer"]),
                    windows=[tuple(map(int, w)) for w in payload["windows"]],
                    from_cache=True,
                )
            except Exception as exc:
                logger.warning("嵌入缓存损坏，重新计算: %s", exc)

        length = len(sequence)
        windows = self.plan_windows(length)
        hidden = self._hidden_size
        accumulator = np.zeros((length, hidden), dtype=np.float64)
        counts = np.zeros(length, dtype=np.float64)

        for start, end in windows:
            sub_sequence = sequence[start:end]
            input_ids = self._encode(sub_sequence).to(self._device)
            outputs = self._forward(input_ids, output_hidden_states=True)
            # hidden_states[-1] 为最后一个编码层输出；去掉 <cls> 与 <eos>
            hidden_states = (
                outputs.hidden_states[-1]
                .float()
                .cpu()
                .numpy()[0, 1 : len(sub_sequence) + 1]
            )
            accumulator[start:end] += hidden_states
            counts[start:end] += 1

        counts[counts == 0] = 1.0
        per_residue = (accumulator / counts[:, None]).astype(np.float32)
        mean = per_residue.mean(axis=0).astype(np.float32)

        result = EmbeddingResult(
            per_residue=per_residue,
            mean=mean,
            layer=-1,
            windows=windows,
            from_cache=False,
        )

        if use_cache:
            try:
                np.savez_compressed(
                    cache_file,
                    per_residue=result.per_residue,
                    mean=result.mean,
                    layer=result.layer,
                    windows=np.array(windows, dtype=np.int64),
                )
            except Exception as exc:
                logger.warning("写入嵌入缓存失败（不影响结果）: %s", exc)

        logger.info("嵌入完成 %s dim=%d windows=%d", describe_sequence(sequence), hidden, len(windows))
        return result

    # ---------------- 掩码边缘打分 ---------------- #
    def masked_marginal(
        self,
        sequence: str,
        positions: list[int] | None = None,
        batch_size: int | None = None,
    ) -> MaskedMarginalResult:
        """逐位点掩码边缘打分（准确通道）。

        Args:
            sequence: 野生型序列。
            positions: 0-based 位点列表；``None`` 表示全部位点。
            batch_size: 批大小，默认取配置 ``embedding_batch_size``。
        """
        self.ensure_loaded()

        import torch

        settings = get_settings()
        batch = batch_size or settings.embedding_batch_size
        length = len(sequence)
        target_positions = sorted(set(range(length) if positions is None else positions))
        target_positions = [p for p in target_positions if 0 <= p < length]
        if not target_positions:
            return MaskedMarginalResult(
                positions=[],
                aa_order=AA_ORDER,
                delta_logprob=np.zeros((0, len(AA_ORDER)), dtype=np.float32),
                wt_logprob=np.zeros((0,), dtype=np.float32),
                windows=[],
            )

        groups = self._position_windows(length, target_positions)
        all_delta: dict[int, np.ndarray] = {}
        all_wt: dict[int, float] = {}
        used_windows: list[tuple[int, int]] = []

        for window_start, group_positions in groups.items():
            window_end = min(window_start + CONTEXT_LIMIT, length)
            used_windows.append((window_start, window_end))
            sub_sequence = sequence[window_start:window_end]
            base_ids = self._encode(sub_sequence)[0]  # [wlen+2]
            mask_id = self._mask_id

            for offset in range(0, len(group_positions), batch):
                chunk = group_positions[offset : offset + batch]
                batch_ids = base_ids.unsqueeze(0).repeat(len(chunk), 1).clone()
                local_indices = []
                for row, position in enumerate(chunk):
                    local = position - window_start
                    local_indices.append(local + 1)
                    batch_ids[row, local + 1] = mask_id

                batch_ids = batch_ids.to(self._device)
                outputs = self._forward(batch_ids)
                logits = outputs.logits.float()  # [B, wlen+2, V]
                position_logits = logits[
                    torch.arange(len(chunk), device=logits.device),
                    torch.tensor(local_indices, device=logits.device),
                    :,
                ]
                log_probs = torch.log_softmax(position_logits, dim=-1)
                aa_scores = log_probs[:, self._aa_ids].cpu().numpy()  # [B, 20]

                for row, position in enumerate(chunk):
                    wild_type = sequence[position]
                    wt_col = AA_ORDER.index(wild_type) if wild_type in AA_ORDER else -1
                    wt_value = float(aa_scores[row, wt_col]) if wt_col >= 0 else float("nan")
                    all_delta[position] = (aa_scores[row] - wt_value).astype(np.float32)
                    all_wt[position] = wt_value

        ordered = sorted(all_delta.keys())
        delta_matrix = np.stack([all_delta[p] for p in ordered]) if ordered else np.zeros((0, 20), np.float32)
        wt_vector = np.array([all_wt[p] for p in ordered], dtype=np.float32)

        logger.info(
            "掩码边缘打分完成 %s 位点数=%d 批大小=%d 窗口数=%d",
            describe_sequence(sequence),
            len(ordered),
            batch,
            len(used_windows),
        )
        return MaskedMarginalResult(
            positions=ordered,
            aa_order=AA_ORDER,
            delta_logprob=delta_matrix,
            wt_logprob=wt_vector,
            windows=used_windows,
        )

    def wild_type_logprobs(self, sequence: str) -> np.ndarray:
        """单次前向获得全部位点的 20 氨基酸对数概率（快速通道）。

        Returns:
            ``[L, 20]`` float32，行对应残基位置，列对应 :data:`AA_ORDER`。
        """
        self.ensure_loaded()

        import torch

        length = len(sequence)
        windows = self.plan_windows(length)
        result = np.full((length, len(AA_ORDER)), np.nan, dtype=np.float32)

        for start, end in windows:
            sub_sequence = sequence[start:end]
            input_ids = self._encode(sub_sequence).to(self._device)
            outputs = self._forward(input_ids)
            log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
            residue_log_probs = log_probs[0, 1 : len(sub_sequence) + 1, :][:, self._aa_ids]
            result[start:end] = residue_log_probs.cpu().numpy()

        return result

    def sequence_naturalness(self, sequence: str) -> float:
        """序列自然度：**掩码条件下**野生型残基的平均对数概率（伪对数似然）。

        为什么必须用掩码版本
        --------------------
        ESM-2 是双向编码器，位置 i 的输出表征通过自注意力**能看到残基 i 自己**。
        因此若直接取未掩码 logits 计算"预测自己"的概率，结果会接近 0（实测某天然
        蛋白为 -0.157），完全失去区分度——模型只是在抄答案。

        正确做法是把每个位置掩掉后读取该位点的对数概率（:meth:`masked_marginal`
        返回的 ``wt_logprob``），这才是蛋白语言模型的标准伪对数似然，天然蛋白
        通常落在 -3 ~ -1 区间，人工设计或不稳定序列显著更低。

        代价：需要 L 次前向（批量执行，实测 382 残基在 V100 上约数秒）。
        """
        import numpy as np

        result = self.masked_marginal(sequence)
        if len(result.wt_logprob) == 0:
            return float("nan")
        values = result.wt_logprob[~np.isnan(result.wt_logprob)]
        if len(values) == 0:
            return float("nan")
        return round(float(np.mean(values)), 4)

    def unload(self) -> None:
        """释放模型显存（运维/多任务场景使用）。"""
        with self._load_lock:
            if not self._loaded:
                return
            try:
                import torch

                del self._model
                self._model = None
                self._tokenizer = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:  # pragma: no cover
                logger.warning("卸载模型时出现异常: %s", exc)
            finally:
                self._loaded = False
                logger.info("ESM-2 模型已卸载")


# --------------------------------------------------------------------------- #
# 进程内单例与便捷函数
# --------------------------------------------------------------------------- #
_service = ESM2Service()


def get_service() -> ESM2Service:
    """获取 ESM-2 服务单例。"""
    return _service


def embedding_status() -> dict[str, Any]:
    """健康检查入口（不加载模型）。"""
    return _service.status()


def embed_sequence(sequence: str, use_cache: bool = True) -> EmbeddingResult:
    """计算序列嵌入。"""
    return _service.embeddings(sequence, use_cache=use_cache)


def masked_marginal_scores(
    sequence: str, positions: list[int] | None = None, batch_size: int | None = None
) -> MaskedMarginalResult:
    """计算掩码边缘打分。"""
    return _service.masked_marginal(sequence, positions=positions, batch_size=batch_size)


def wild_type_scores(sequence: str) -> np.ndarray:
    """计算快速通道全位点对数概率。"""
    return _service.wild_type_logprobs(sequence)


def sequence_naturalness(sequence: str) -> float:
    """序列自然度（越接近 0 越自然）。"""
    return _service.sequence_naturalness(sequence)


def unload_model() -> None:
    """释放模型。"""
    _service.unload()
