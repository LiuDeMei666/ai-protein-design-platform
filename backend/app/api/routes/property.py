"""理化性质预测路由。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ...core.errors import NotFoundError
from ...db.models import ProteinSequence, PropertyRun
from ...jobs.queue import submit_job
from ...jobs.runner import HEAVY_KINDS, handler_for
from ...schemas.common import JobAccepted, Page
from ...schemas.property import PropertyPredictIn
from ...services.property.metric import METRIC_LABELS, RISK_LABELS
from ...services.property.pipeline import METRIC_ORDER
from ..deps import get_db, resolve_project_id

router = APIRouter(prefix="/property", tags=["property"])


@router.get("/catalog", summary="性质指标说明")
def catalog() -> dict[str, Any]:
    """返回 9 个指标的键、中文标签与算法来源，供前端渲染说明与生成文档。"""
    return {
        "metrics": [
            {"key": key, "label": METRIC_LABELS.get(key, key)} for key in METRIC_ORDER
        ],
        "risk_labels": RISK_LABELS,
        "note": (
            "所有指标均为 0-100 分，分数越高越好（风险类指标为安全性得分）。"
            "每项都携带算法来源与逐条证据，可在报告中逐条核对。"
        ),
    }


@router.post("/predict", response_model=JobAccepted, summary="提交性质预测作业")
def predict(payload: PropertyPredictIn, session: Session = Depends(get_db)) -> JobAccepted:
    """提交异步性质预测作业。

    结构预测为可选增强项：即使断网导致结构预测失败，仍会给出完整的 9 项指标
    （相应证据项标注为"不可用"并在权重中剔除）。
    """
    project_id = resolve_project_id(session, payload.project_id)
    params = {
        "sequence": payload.sequence,
        "protein_type": payload.protein_type,
        "host_system": payload.host_system,
        "use_structure": payload.use_structure,
        "provider": payload.provider,
        "use_esm": payload.use_esm,
        "dna_sequence": payload.dna_sequence,
        "name": payload.name,
        "project_id": project_id,
    }
    job_id = submit_job(
        "property",
        handler_for("property"),
        params,
        project_id=project_id,
        title=f"性质预测（{payload.protein_type}）",
        heavy="property" in HEAVY_KINDS,
    )
    return JobAccepted(job_id=job_id)


@router.get("/runs", response_model=Page[dict], summary="列出历史性质预测")
def list_runs(
    sequence_id: int | None = None,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
) -> Page[dict]:
    query = session.query(PropertyRun)
    if sequence_id is not None:
        query = query.filter(PropertyRun.sequence_id == sequence_id)
    total = query.count()
    items = query.order_by(PropertyRun.id.desc()).offset(offset).limit(limit).all()

    payload: list[dict] = []
    for record in items:
        sequence = session.get(ProteinSequence, record.sequence_id)
        payload.append(
            {
                "id": record.id,
                "sequence_id": record.sequence_id,
                "sequence_name": sequence.name if sequence else None,
                "sequence_length": sequence.length if sequence else None,
                "host_system": record.host_system,
                "use_structure": record.use_structure,
                "overall_score": (record.summary or {}).get("overall_score"),
                "risk_counts": (record.summary or {}).get("risk_counts", {}),
                "created_at": record.created_at,
            }
        )
    return Page[dict](items=payload, total=total, limit=limit, offset=offset)


@router.get("/runs/{run_id}", summary="性质预测详情")
def get_run(run_id: int, session: Session = Depends(get_db)) -> dict[str, Any]:
    record = session.get(PropertyRun, run_id)
    if record is None:
        raise NotFoundError(f"性质预测记录 id={run_id} 不存在")
    sequence = session.get(ProteinSequence, record.sequence_id)
    payload = dict(record.results or {})
    payload["run_id"] = record.id
    payload["sequence_id"] = record.sequence_id
    payload["sequence_name"] = sequence.name if sequence else None
    payload["created_at"] = record.created_at
    return payload


@router.delete("/runs/{run_id}", summary="删除性质预测记录")
def delete_run(run_id: int, session: Session = Depends(get_db)) -> dict[str, Any]:
    record = session.get(PropertyRun, run_id)
    if record is None:
        raise NotFoundError(f"性质预测记录 id={run_id} 不存在")
    session.delete(record)
    return {"ok": True, "message": f"已删除记录 {run_id}"}
