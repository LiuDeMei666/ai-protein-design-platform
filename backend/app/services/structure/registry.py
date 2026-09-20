"""Provider 注册表：选择、降级与磁盘缓存。

降级链（``structure_provider=auto`` 时）::

    esmatlas（在线，优先）
        └─ 失败/不可用 ─> local_esmfold（离线后备，需装 fair-esm）
                └─ 失败/不可用 ─> stub（占位，保证平台可用）

**任何一次降级都会写入** :attr:`StructureResult.degradation_reason`，并在
:attr:`StructureResult.warnings` 里给出中文说明，前端必须展示。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from ...core.config import get_settings
from ...core.errors import PlatformError, ProviderUnavailableError, ValidationError
from ...core.logging import describe_sequence, get_logger, sequence_fingerprint
from .base import BasePredictor, StructureResult
from .esmatlas import ESMAtlasPredictor
from .local_esmfold import LocalESMFoldPredictor
from .stub import StubPredictor

logger = get_logger(__name__)

#: auto 模式下的尝试顺序
AUTO_ORDER: tuple[str, ...] = ("esmatlas", "local_esmfold", "stub")

_providers: dict[str, BasePredictor] | None = None
_registry_lock = threading.RLock()
_cache_lock = threading.RLock()


def get_providers() -> dict[str, BasePredictor]:
    """返回全部 Provider（惰性构建，进程内复用）。"""
    global _providers
    if _providers is None:
        with _registry_lock:
            if _providers is None:
                _providers = {
                    "esmatlas": ESMAtlasPredictor(),
                    "local_esmfold": LocalESMFoldPredictor(),
                    "stub": StubPredictor(),
                }
    return _providers


def get_provider(name: str) -> BasePredictor:
    """按名称取 Provider。"""
    providers = get_providers()
    if name not in providers:
        raise ValidationError(
            f"未知的结构预测 Provider: {name}",
            detail={"available": sorted(providers.keys())},
        )
    return providers[name]


def provider_status() -> dict[str, dict[str, Any]]:
    """全部 Provider 的状态快照（供健康检查）。"""
    status: dict[str, dict[str, Any]] = {}
    for name, provider in get_providers().items():
        try:
            status[name] = provider.describe()
        except Exception as exc:  # pragma: no cover - 防御
            status[name] = {"name": name, "available": False, "reason": str(exc)}
    return status


def resolve_chain(preferred: str | None = None) -> list[BasePredictor]:
    """计算本次要尝试的 Provider 顺序。

    * ``preferred`` 为空或 ``auto`` -> 按 :data:`AUTO_ORDER`。
    * 指定具体 Provider -> 该 Provider 优先，其后补齐其它可用者作为降级。
    """
    providers = get_providers()
    settings = get_settings()
    requested = preferred or settings.structure_provider

    if requested in (None, "", "auto"):
        order = list(AUTO_ORDER)
    else:
        if requested not in providers:
            raise ValidationError(
                f"未知的结构预测 Provider: {requested}",
                detail={"available": sorted(providers.keys())},
            )
        order = [requested] + [name for name in AUTO_ORDER if name != requested]

    return [providers[name] for name in order]


# --------------------------------------------------------------------------- #
# 磁盘缓存
# --------------------------------------------------------------------------- #
def _cache_file(sequence: str, provider_name: str) -> Path:
    settings = get_settings()
    directory = Path(settings.cache_dir) / "structures" / provider_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{sequence_fingerprint(sequence)}_{len(sequence)}.json"


def _save_cache(sequence: str, result: StructureResult) -> None:
    path = _cache_file(sequence, result.source)
    payload = {
        "pdb_text": result.pdb_text,
        "plddt": result.plddt,
        "mean_plddt": result.mean_plddt,
        "source": result.source,
        # 降级原因必须一并持久化：它记录着"这份结构是不是占位骨架"。
        # 早期版本漏存该字段，导致缓存命中后 degradation_reason 恒为 None，
        # 同一条序列首次调用与二次调用返回的原因不一致，
        # 数据库里也出现"provider=stub 却显示无降级"的记录。
        "degradation_reason": result.degradation_reason,
        "truncated": result.truncated,
        "segments": [list(segment) for segment in result.segments],
        "model_version": result.model_version,
        "stats": result.stats,
        "warnings": result.warnings,
    }
    with _cache_lock:
        try:
            # 原子写入：先写临时文件再替换，避免并发读到半截 JSON
            temp_path = path.with_suffix(".json.tmp")
            temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(temp_path, path)
        except Exception as exc:
            logger.warning("写入结构缓存失败（不影响本次结果）: %s", exc)


def _load_cache(sequence: str, provider_name: str) -> StructureResult | None:
    path = _cache_file(sequence, provider_name)
    if not path.exists():
        return None
    with _cache_lock:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("结构缓存损坏，忽略: %s", exc)
            try:
                path.unlink()
            except OSError:
                pass
            return None

    return StructureResult(
        pdb_text=payload["pdb_text"],
        plddt=payload["plddt"],
        mean_plddt=payload["mean_plddt"],
        source=payload["source"],
        truncated=payload.get("truncated", False),
        segments=[tuple(segment) for segment in payload.get("segments", [])],
        from_cache=True,
        # 从缓存恢复降级原因（旧缓存文件没有该字段，.get 会回落到 None，
        # 属可接受的向后兼容；清一次缓存即可让旧记录补齐）
        degradation_reason=payload.get("degradation_reason"),
        model_version=payload.get("model_version"),
        stats=payload.get("stats", {}),
        warnings=payload.get("warnings", []),
    )


def clear_cache(sequence: str | None = None) -> int:
    """清理结构缓存。``sequence`` 为空时清空全部，返回删除文件数。"""
    settings = get_settings()
    root = Path(settings.cache_dir) / "structures"
    if not root.exists():
        return 0
    removed = 0
    for path in root.rglob("*.json"):
        if sequence is not None and sequence_fingerprint(sequence) not in path.name:
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def predict_structure(
    sequence: str,
    *,
    provider: str | None = None,
    allow_split: bool = True,
    use_cache: bool = True,
) -> StructureResult:
    """预测结构，带缓存与自动降级。

    Args:
        sequence: 目标序列。
        provider: 指定 Provider；``None`` / ``auto`` 时按降级链自动选择。
        allow_split: 超长序列是否允许分片折叠。
        use_cache: 是否使用磁盘缓存。

    Raises:
        ProviderUnavailableError: 所有 Provider 均失败（含 stub，通常不会发生）。
    """
    if not sequence:
        raise ValidationError("序列为空，无法进行结构预测")

    chain = resolve_chain(provider)
    settings = get_settings()

    # 1) 命中缓存直接返回（按降级链顺序查找，优先返回更高优先级 Provider 的缓存）
    if use_cache and settings.structure_provider in ("auto", None, "") and provider in (None, "", "auto"):
        for candidate in chain:
            cached = _load_cache(sequence, candidate.name)
            if cached is not None:
                logger.info("结构缓存命中 %s provider=%s", describe_sequence(sequence), candidate.name)
                return cached

    # 2) 依次尝试
    failures: list[str] = []
    for index, candidate in enumerate(chain):
        is_fallback = index > 0
        try:
            if not candidate.available():
                failures.append(f"{candidate.name}: 当前环境不可用")
                continue

            result = candidate.predict(sequence, allow_split=allow_split)

            if is_fallback:
                reason = (
                    f"首选 Provider 不可用或失败（{' ; '.join(failures)}），"
                    f"已降级为 {candidate.name}。"
                )
                result.degradation_reason = reason
                result.warnings = [reason, *result.warnings]
                logger.warning("结构预测降级: %s", reason)

            # 降级结果**不得写入缓存**
            # ----------------------
            # 缓存的意义是"避免重复计算昂贵且可复现的结果"。降级到 stub 是
            # **失败后的占位**：既没有真正算过，也谈不上可复现（网络一恢复就该重算）。
            # 若把它当成功结果缓存，会带来一个很难察觉的后果——缓存查找按降级链顺序
            # 遍历（esmatlas -> local_esmfold -> stub），stub 的缓存一旦命中就短路返回，
            # 于是**一次瞬时故障会把这条序列永久钉在假结构上**，此后再也不重试真实通道。
            # 实测：COL1A1 因一次 504 生成 stub 缓存后，普通请求 0.01s 返回 pLDDT 37.61 的
            # 假骨架，而强制走在线通道本可得到 pLDDT 70.25 的真实结构。
            if use_cache and not result.degradation_reason:
                _save_cache(sequence, result)
            return result

        except PlatformError as exc:
            failures.append(f"{candidate.name}: {exc.message}")
            logger.warning("Provider %s 失败: %s", candidate.name, exc.message)
            continue
        except Exception as exc:  # 未预期异常不应中断降级链
            failures.append(f"{candidate.name}: {exc}")
            logger.exception("Provider %s 抛出未预期异常", candidate.name)
            continue

    raise ProviderUnavailableError(
        "所有结构预测通道均失败，请检查网络或部署本地 ESMFold",
        detail={"attempts": failures, "hint": "在线通道需要外网；离线请安装 fair-esm"},
    )


def reset_providers() -> None:
    """测试用：清空 Provider 缓存。"""
    global _providers
    with _registry_lock:
        _providers = None
