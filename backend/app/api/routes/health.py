"""健康检查：外部依赖、模型与存储状态一览。

设计原则：健康检查**不加载重型模型**，只报告"可用性"与"是否已就绪"，
避免一次探活就把 V100 显存占满。
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from ...core.config import Settings
from ...core.logging import get_logger
from ...db.base import get_engine
from ..deps import get_config

router = APIRouter(tags=["health"])
logger = get_logger(__name__)


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:  # 并发删除等竞态，忽略
            continue
    return round(total / 1024 / 1024, 2)


def _torch_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - 依赖缺失时的兜底
        return {"installed": False, "error": str(exc)}
    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def _structure_status() -> dict[str, Any]:
    """结构预测 Provider 状态；模块尚未落地时不报错，标记为 unavailable。"""
    try:
        from ...services.structure.registry import provider_status
    except Exception as exc:
        return {"ok": False, "reason": f"模块未就绪: {exc}"}
    try:
        return {"ok": True, "providers": provider_status()}
    except Exception as exc:
        logger.warning("结构 Provider 状态检查失败: %s", exc)
        return {"ok": False, "reason": str(exc)}


def _embedding_status() -> dict[str, Any]:
    """ESM-2 权重与设备状态（不加载模型本体）。"""
    try:
        from ...services.embedding.esm2 import embedding_status
    except Exception as exc:
        return {"ok": False, "reason": f"模块未就绪: {exc}"}
    try:
        return {"ok": True, **embedding_status()}
    except Exception as exc:
        logger.warning("嵌入服务状态检查失败: %s", exc)
        return {"ok": False, "reason": str(exc)}


def _db_status() -> dict[str, Any]:
    try:
        settings = get_config()
        engine = get_engine()
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        size_mb = (
            round(settings.db_path.stat().st_size / 1024 / 1024, 2)
            if settings.db_path.exists()
            else 0.0
        )
        return {"ok": True, "path": str(settings.db_path), "size_mb": size_mb}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


@router.get("/health", summary="健康检查")
def health(settings: Settings = Depends(get_config)) -> dict[str, Any]:
    """平台自检：数据库、缓存、GPU、结构预测与嵌入服务。"""
    db = _db_status()
    structure = _structure_status()
    embedding = _embedding_status()

    # 核心可用性：数据库通 + 至少一个结构 Provider 可用
    providers = structure.get("providers", {}) if structure.get("ok") else {}
    any_provider = (
        any(item.get("available") for item in providers.values()) if providers else False
    )

    return {
        "app": settings.app_name,
        "version": settings.version,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "database": db,
        "torch": _torch_status(),
        "structure": structure,
        "embedding": embedding,
        "storage": {
            "cache_dir": str(settings.cache_dir),
            "cache_size_mb": _dir_size_mb(settings.cache_dir),
            "models_dir": str(settings.models_dir),
            "models_size_mb": _dir_size_mb(settings.models_dir),
            "logs_dir": str(settings.logs_dir),
        },
        "config": {
            "structure_provider": settings.structure_provider,
            "esm_model": settings.esm_model,
            "hf_endpoint": settings.hf_endpoint,
            "device": settings.device,
            "use_fp16": settings.use_fp16,
            "job_workers": settings.job_workers,
            "api_key_required": settings.require_api_key,
        },
        "healthy": bool(db.get("ok")) and any_provider,
    }


@router.get("/health/live", summary="存活探针")
def liveness() -> dict[str, str]:
    """仅表示进程存活，用于容器编排。"""
    return {"status": "alive"}
