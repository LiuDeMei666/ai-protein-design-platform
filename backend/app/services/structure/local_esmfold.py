"""本地 ESMFold Provider（离线后备）。

状态说明
--------
本机 **尚未安装** ``fair-esm`` 与 ``openfold``，因此 :meth:`available` 默认返回
``False``，注册表会自动跳过它。安装方式::

    pip install fair-esm
    # openfold 需要按官方仓库单独安装（编译依赖较重）

部署价值
--------
企业内网完全离线时，这是唯一能产出**真实**结构的通道。V100 32GB 可运行
``esmfold_3B_v1``（约 15GB 权重，fp16 推理）。依赖缺失时安全禁用，绝不阻断平台启动。
"""

from __future__ import annotations

import threading
import time
from typing import Any

from ...core.config import get_settings
from ...core.errors import ExternalServiceError, ModelNotAvailableError
from ...core.logging import describe_sequence, get_logger
from . import pdb_utils
from .base import BasePredictor, StructureResult

logger = get_logger(__name__)


class LocalESMFoldPredictor(BasePredictor):
    """本地 ESMFold 推理（惰性加载权重）。"""

    name = "local_esmfold"
    max_length: int | None = 1024  # ESMFold 与 ESM-2 共享位置上限

    def __init__(self, weights: str | None = None) -> None:
        settings = get_settings()
        self.weights = weights or settings.local_esmfold_weights
        self.model_version = f"ESMFold {self.weights}"
        self._model: Any = None
        self._lock = threading.RLock()
        self._load_error: str | None = None

    # ------------------------------------------------------------------ #
    def _import_stack(self) -> tuple[Any, Any]:
        """导入 esm + openfold，返回 (esm 模块, torch)。缺失时抛异常。"""
        try:
            import esm  # type: ignore
        except ImportError as exc:
            raise ModelNotAvailableError(
                "未安装 fair-esm，无法使用本地 ESMFold",
                detail={"install": "pip install fair-esm", "error": str(exc)},
            ) from exc
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ModelNotAvailableError(f"未安装 torch: {exc}") from exc
        return esm, torch

    def available(self) -> bool:
        """检查依赖是否可导入（不加载权重）。"""
        if self._load_error is not None:
            return False
        try:
            self._import_stack()
            return True
        except Exception:
            return False

    def describe(self) -> dict[str, Any]:
        available = self.available()
        detail: dict[str, Any] = {
            "name": self.name,
            "available": available,
            "max_length": self.max_length,
            "model_version": self.model_version,
            "weights": self.weights,
            "loaded": self._model is not None,
        }
        if not available:
            detail["reason"] = "未安装 fair-esm / openfold"
            detail["install"] = "pip install fair-esm（openfold 需按官方仓库安装）"
        if self._load_error:
            detail["load_error"] = self._load_error
        return detail

    # ------------------------------------------------------------------ #
    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            esm, torch = self._import_stack()
            logger.info("加载本地 ESMFold 权重: %s", self.weights)
            started = time.perf_counter()
            try:
                model = esm.pretrained.esmfold_v1()
                model = model.eval()
                if torch.cuda.is_available():
                    model = model.cuda()
                    if get_settings().use_fp16:
                        model = model.half()
                self._model = model
            except Exception as exc:
                self._load_error = str(exc)
                raise ModelNotAvailableError(
                    f"本地 ESMFold 加载失败: {exc}",
                    detail={"weights": self.weights},
                ) from exc
            logger.info("本地 ESMFold 就绪，耗时 %.1fs", time.perf_counter() - started)
            return self._model

    def predict(self, sequence: str, *, allow_split: bool = True) -> StructureResult:
        sequence = self._require_sequence(sequence)
        if self.max_length and len(sequence) > self.max_length:
            raise ExternalServiceError(
                f"序列长度 {len(sequence)} 超过本地 ESMFold 上限 {self.max_length}",
                detail={"hint": "请先按结构域拆分，或改用分片策略"},
            )

        model = self._load()
        _, torch = self._import_stack()

        logger.info("本地 ESMFold 折叠 %s", describe_sequence(sequence))
        started = time.perf_counter()
        try:
            with torch.no_grad():
                pdb_text = model.infer_pdb(sequence)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise ExternalServiceError(
                    "GPU 显存不足，本地 ESMFold 推理失败",
                    detail={"error": str(exc), "hint": "尝试缩短序列或改用 CPU 模式"},
                ) from exc
            raise ExternalServiceError(f"本地 ESMFold 推理失败: {exc}") from exc
        elapsed = time.perf_counter() - started

        data = pdb_utils.parse_pdb(pdb_text, source=self.name)
        plddt = pdb_utils.extract_plddt(data)
        stats = pdb_utils.build_structure_stats(data, plddt)
        stats["elapsed_seconds"] = round(elapsed, 2)

        logger.info(
            "本地 ESMFold 完成 残基=%d 耗时=%.2fs", data.length, elapsed
        )

        return StructureResult(
            pdb_text=pdb_text,
            plddt=plddt,
            mean_plddt=round(float(sum(plddt) / len(plddt)), 2) if plddt else 0.0,
            source=self.name,
            truncated=False,
            segments=[(0, data.length)],
            model_version=self.model_version,
            stats=stats,
        )

    def unload(self) -> None:
        """释放显存。"""
        with self._lock:
            if self._model is None:
                return
            try:
                import torch

                del self._model
                self._model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:  # pragma: no cover
                logger.warning("卸载本地 ESMFold 异常: %s", exc)
