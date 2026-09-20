"""结构预测路由。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from ...core.errors import NotFoundError
from ...db.models import StructureRecord
from ...jobs.queue import submit_job
from ...jobs.runner import HEAVY_KINDS, handler_for
from ...schemas.common import JobAccepted, Page
from ...schemas.structure import ProviderStatusOut, StructurePredictIn
from ...services.structure.registry import clear_cache, provider_status
from ..deps import get_db, resolve_project_id

router = APIRouter(prefix="/structure", tags=["structure"])


@router.get("/providers", response_model=list[ProviderStatusOut], summary="结构预测通道状态")
def providers() -> list[ProviderStatusOut]:
    """返回各 Provider 的可用性、长度上限与降级说明。

    前端应在提交前用它提示用户"当前是否依赖外网"。
    """
    payload: list[ProviderStatusOut] = []
    for name, detail in provider_status().items():
        payload.append(
            ProviderStatusOut(
                name=name,
                available=bool(detail.get("available")),
                max_length=detail.get("max_length"),
                model_version=detail.get("model_version"),
                note=detail.get("note"),
                reason=detail.get("reason"),
                install=detail.get("install"),
            )
        )
    return payload


@router.post("/predict", response_model=JobAccepted, summary="提交结构预测作业")
def predict(payload: StructurePredictIn, session: Session = Depends(get_db)) -> JobAccepted:
    """提交异步结构预测作业，返回 job_id。

    ESM Atlas 对 400 残基以内的序列通常 2-30 秒，接近上限的序列可能需要数分钟。
    超长序列会自动分片折叠，并在结果中标注分片边界。
    """
    params = {
        "sequence": payload.sequence,
        "protein_type": payload.protein_type,
        "provider": payload.provider,
        "allow_split": payload.allow_split,
        "use_cache": payload.use_cache,
        "name": payload.name,
        "project_id": resolve_project_id(session, payload.project_id),
    }
    job_id = submit_job(
        "structure",
        handler_for("structure"),
        params,
        project_id=params["project_id"],
        title=f"结构预测（{len(payload.sequence)} aa）",
        heavy="structure" in HEAVY_KINDS,
    )
    return JobAccepted(job_id=job_id)


@router.get("/records", response_model=Page[dict], summary="列出已缓存的结构")
def list_structures(
    sequence_sha256: str | None = None,
    limit: int = Query(default=30, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
) -> Page[dict]:
    query = session.query(StructureRecord)
    if sequence_sha256:
        query = query.filter(StructureRecord.sequence_sha256 == sequence_sha256)
    total = query.count()
    items = (
        query.order_by(StructureRecord.id.desc()).offset(offset).limit(limit).all()
    )
    return Page[dict](
        items=[
            {
                "id": record.id,
                "sequence_sha256": record.sequence_sha256,
                "sequence_length": record.sequence_length,
                "provider": record.provider,
                "mean_plddt": record.mean_plddt,
                "truncated": record.truncated,
                "segments": record.segments,
                "degradation_reason": record.degradation_reason,
                "created_at": record.created_at,
            }
            for record in items
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{structure_id}/pdb", response_class=PlainTextResponse, summary="获取 PDB 文本")
def get_pdb(structure_id: int, session: Session = Depends(get_db)) -> PlainTextResponse:
    """返回 PDB 文本，供 3Dmol.js 渲染。

    单独提供此接口，是为了让作业结果里不必携带数百 KB 的文本。
    """
    record = session.get(StructureRecord, structure_id)
    if record is None:
        raise NotFoundError(f"结构 id={structure_id} 不存在")
    from pathlib import Path

    path = Path(record.pdb_path)
    if not path.exists():
        raise NotFoundError(f"结构文件缺失：{record.pdb_path}（缓存可能已被清理）")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="chemical/x-pdb")


@router.get("/{structure_id}", summary="结构元数据")
def get_structure(structure_id: int, session: Session = Depends(get_db)) -> dict[str, Any]:
    record = session.get(StructureRecord, structure_id)
    if record is None:
        raise NotFoundError(f"结构 id={structure_id} 不存在")
    return {
        "id": record.id,
        "sequence_sha256": record.sequence_sha256,
        "sequence_length": record.sequence_length,
        "provider": record.provider,
        "model_version": record.model_version,
        "mean_plddt": record.mean_plddt,
        "plddt": record.plddt,
        "truncated": record.truncated,
        "segments": record.segments,
        "secondary_structure": record.secondary_structure,
        "stats": record.stats,
        "degradation_reason": record.degradation_reason,
        "pdb_url": f"/api/structure/{record.id}/pdb",
        "created_at": record.created_at,
    }


@router.delete("/cache", summary="清理结构缓存")
def purge_cache(sequence: str | None = None) -> dict[str, Any]:
    """清理磁盘结构缓存（不影响数据库记录）。"""
    removed = clear_cache(sequence)
    return {"removed_files": removed, "scope": "单序列" if sequence else "全部"}
