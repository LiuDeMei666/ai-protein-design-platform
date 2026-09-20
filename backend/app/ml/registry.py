"""模型版本注册与切换。

需求文档要求"模型随使用不断进化"，因此必须有版本管理：
* 每次训练/增量训练都产生一个**新版本**，旧版本保留可回溯；
* 每个属性同一时刻只有一个 ``is_active`` 版本，预测时默认使用它；
* 支持一键回滚到历史版本（把 ``is_active`` 切回去即可，工件仍在本机）。
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy.orm import Session

from ..core.errors import NotFoundError
from ..core.logging import get_logger
from ..db.models import ModelVersion

logger = get_logger(__name__)

#: 模型工件在 models/ 下的子目录
ALGO_ARTIFACT_DIR = "property_heads"

_VERSION_PATTERN = re.compile(r"^v(\d+)$")


def next_version_label(session: Session, property_name: str) -> str:
    """生成下一个版本号（``v1``、``v2``…）。"""
    existing = (
        session.query(ModelVersion.version)
        .filter(ModelVersion.property_name == property_name)
        .all()
    )
    maximum = 0
    for (label,) in existing:
        match = _VERSION_PATTERN.match(str(label))
        if match:
            maximum = max(maximum, int(match.group(1)))
    return f"v{maximum + 1}"


def get_active_version(session: Session, property_name: str) -> ModelVersion | None:
    """取某属性的激活版本。"""
    return (
        session.query(ModelVersion)
        .filter(ModelVersion.property_name == property_name, ModelVersion.is_active.is_(True))
        .order_by(ModelVersion.created_at.desc())
        .first()
    )


def set_active_version(session: Session, record: ModelVersion) -> ModelVersion:
    """把指定版本设为激活，同时取消同属性其它版本的激活状态。"""
    session.query(ModelVersion).filter(
        ModelVersion.property_name == record.property_name,
        ModelVersion.id != record.id,
    ).update({"is_active": False})
    record.is_active = True
    session.flush()
    logger.info("模型激活：属性=%s 版本=%s", record.property_name, record.version)
    return record


def activate_by_id(session: Session, version_id: int) -> ModelVersion:
    """按 id 激活某个版本（回滚用）。"""
    record = session.get(ModelVersion, version_id)
    if record is None:
        raise NotFoundError(f"模型版本 id={version_id} 不存在")
    if record.status != "ready":
        from ..core.errors import ValidationError

        raise ValidationError(
            f"版本 {record.version} 状态为 {record.status}，不可激活",
            detail={"status": record.status},
        )
    return set_active_version(session, record)


def list_versions(
    session: Session, property_name: str | None = None, limit: int = 100
) -> list[ModelVersion]:
    """列出模型版本（按创建时间倒序）。"""
    query = session.query(ModelVersion)
    if property_name:
        query = query.filter(ModelVersion.property_name == property_name)
    return query.order_by(ModelVersion.created_at.desc()).limit(limit).all()


def to_payload(record: ModelVersion) -> dict[str, Any]:
    """ORM -> 响应字典。"""
    return {
        "id": record.id,
        "property_name": record.property_name,
        "version": record.version,
        "algo": record.algo,
        "base_model": record.base_model,
        "n_samples": record.n_samples,
        "n_features": record.n_features,
        "metrics": record.metrics or {},
        "status": record.status,
        "is_active": record.is_active,
        "note": record.note,
        "created_at": record.created_at,
        "artifact_path": record.artifact_path,
    }
