"""模型版本与增量训练路由。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ...core.errors import NotFoundError
from ...jobs.queue import submit_job
from ...jobs.runner import handler_for
from ...ml.datasets import available_properties
from ...ml.registry import activate_by_id, get_active_version, list_versions, to_payload
from ...schemas.common import JobAccepted, OkOut, Page
from ...schemas.experiment import ModelVersionOut, TrainRequestIn
from ..deps import get_db

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/versions", response_model=Page[ModelVersionOut], summary="列出模型版本")
def versions(
    property_name: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
) -> Page[ModelVersionOut]:
    """列出属性头的全部版本（含指标、样本量与激活状态）。"""
    all_versions = list_versions(session, property_name, limit=1000)
    total = len(all_versions)
    page = all_versions[offset : offset + limit]
    return Page[ModelVersionOut](
        items=[ModelVersionOut(**to_payload(item)) for item in page],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/active", summary="当前激活的模型")
def active(session: Session = Depends(get_db)) -> dict[str, Any]:
    """列出每个属性当前激活的版本——前端状态栏据此显示"模型版本"。"""
    from sqlalchemy import distinct

    from ...db.models import ModelVersion

    names = [
        row[0]
        for row in session.query(distinct(ModelVersion.property_name)).all()
    ]
    payload: dict[str, Any] = {}
    for name in names:
        record = get_active_version(session, name)
        if record is not None:
            payload[name] = to_payload(record)
    return {"active_models": payload, "count": len(payload)}


@router.get("/properties", summary="可训练的属性与样本量")
def properties(
    min_records: int = Query(default=2, ge=1),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    """列出已有实验数据的属性及其样本量，用于判断"现在能不能训练"。"""
    return {
        "properties": available_properties(session, min_records),
        "note": "建议每个属性至少积累 15 条以上配对数据再训练，否则评估指标不具备参考价值。",
    }


@router.post("/train", response_model=JobAccepted, summary="提交训练作业")
def train(payload: TrainRequestIn, session: Session = Depends(get_db)) -> JobAccepted:
    """提交属性头训练作业（默认全量训练）。

    训练完成后新版本自动激活；历史版本保留，可随时回滚。
    """
    job_id = submit_job(
        "train",
        handler_for("train"),
        {
            "property_name": payload.property_name,
            "algo": payload.algo,
            "min_samples": payload.min_samples,
            "test_ratio": payload.test_ratio,
            "cv_folds": payload.cv_folds,
            "notes": payload.notes,
        },
        title=f"训练属性头（{payload.property_name}）",
        heavy=False,
    )
    return JobAccepted(job_id=job_id)


@router.post("/incremental", response_model=JobAccepted, summary="提交增量训练作业")
def incremental(
    property_name: str = Query(description="目标属性"),
    session: Session = Depends(get_db),
) -> JobAccepted:
    """在激活版本基础上做增量训练，只消费水位线之后的新增实验记录。"""
    job_id = submit_job(
        "train",
        handler_for("train"),
        {"property_name": property_name, "incremental": True},
        title=f"增量训练（{property_name}）",
        heavy=False,
    )
    return JobAccepted(job_id=job_id)


@router.post("/versions/{version_id}/activate", response_model=OkOut, summary="激活指定版本")
def activate(version_id: int, session: Session = Depends(get_db)) -> OkOut:
    """回滚：把某个历史版本重新设为激活。"""
    from ...db.models import ModelVersion

    record = session.get(ModelVersion, version_id)
    if record is None:
        raise NotFoundError(f"模型版本 id={version_id} 不存在")
    activate_by_id(session, version_id)
    return OkOut(message=f"已激活 {record.property_name} 版本 {record.version}")


@router.get("/versions/{version_id}", summary="版本详情")
def version_detail(version_id: int, session: Session = Depends(get_db)) -> dict[str, Any]:
    from ...db.models import ModelVersion

    record = session.get(ModelVersion, version_id)
    if record is None:
        raise NotFoundError(f"模型版本 id={version_id} 不存在")
    return to_payload(record)
