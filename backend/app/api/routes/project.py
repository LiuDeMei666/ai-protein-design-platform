"""项目与序列管理路由。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from ...core.errors import NotFoundError, ValidationError
from ...db.models import DesignRun, ExperimentRecord, Project, ProteinSequence
from ...schemas.common import OkOut, Page
from ...schemas.sequence import (
    PROTEIN_TYPE_LABELS,
    ProjectCreateIn,
    ProjectOut,
    SequenceCreateIn,
    SequenceOut,
)
from ...services.sequence.validator import validate_sequence
from ..deps import get_db

router = APIRouter(tags=["project"])


def _project_out(session: Session, project: Project) -> ProjectOut:
    sequence_count = (
        session.query(func.count(ProteinSequence.id))
        .filter(ProteinSequence.project_id == project.id)
        .scalar()
        or 0
    )
    design_count = (
        session.query(func.count(DesignRun.id))
        .join(ProteinSequence, DesignRun.sequence_id == ProteinSequence.id)
        .filter(ProteinSequence.project_id == project.id)
        .scalar()
        or 0
    )
    experiment_count = (
        session.query(func.count(ExperimentRecord.id))
        .filter(ExperimentRecord.project_id == project.id)
        .scalar()
        or 0
    )
    return ProjectOut(
        id=project.id,
        name=project.name,
        description=project.description,
        host_system=project.host_system,
        created_at=project.created_at,
        sequence_count=int(sequence_count),
        design_count=int(design_count),
        experiment_count=int(experiment_count),
    )


@router.get("/projects", response_model=Page[ProjectOut], summary="列出项目")
def list_projects(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
) -> Page[ProjectOut]:
    query = session.query(Project).order_by(Project.id)
    total = query.count()
    items = query.offset(offset).limit(limit).all()
    return Page[ProjectOut](
        items=[_project_out(session, item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/projects", response_model=ProjectOut, summary="创建项目")
def create_project(payload: ProjectCreateIn, session: Session = Depends(get_db)) -> ProjectOut:
    exists = session.query(Project).filter(Project.name == payload.name).one_or_none()
    if exists is not None:
        raise ValidationError(f"项目名 {payload.name} 已存在", detail={"project_id": exists.id})
    project = Project(
        name=payload.name,
        description=payload.description,
        host_system=payload.host_system,
    )
    session.add(project)
    session.flush()
    return _project_out(session, project)


@router.get("/projects/{project_id}", response_model=ProjectOut, summary="项目详情")
def get_project(project_id: int, session: Session = Depends(get_db)) -> ProjectOut:
    project = session.get(Project, project_id)
    if project is None:
        raise NotFoundError(f"项目 id={project_id} 不存在")
    return _project_out(session, project)


@router.delete("/projects/{project_id}", response_model=OkOut, summary="删除项目")
def delete_project(project_id: int, session: Session = Depends(get_db)) -> OkOut:
    """级联删除项目下的序列、设计批次与实验记录。"""
    project = session.get(Project, project_id)
    if project is None:
        raise NotFoundError(f"项目 id={project_id} 不存在")
    if project.name == "默认项目":
        raise ValidationError("默认项目不可删除（平台初始化依赖它）")
    session.delete(project)
    return OkOut(message=f"已删除项目 {project.name}")


# --------------------------------------------------------------------------- #
# 序列
# --------------------------------------------------------------------------- #
def _sequence_out(session: Session, record: ProteinSequence, include_sequence: bool = True) -> SequenceOut:
    protein_type = record.protein_type or "generic"
    return SequenceOut(
        id=record.id,
        project_id=record.project_id,
        name=record.name,
        protein_type=protein_type,
        protein_type_label=PROTEIN_TYPE_LABELS.get(protein_type, protein_type),
        sequence=record.sequence if include_sequence else "",
        length=record.length,
        sha256=record.sha256,
        note=record.note,
        created_at=record.created_at,
        extra={
            "design_count": session.query(func.count(DesignRun.id))
            .filter(DesignRun.sequence_id == record.id)
            .scalar()
            or 0,
        },
    )


@router.get("/sequences", response_model=Page[SequenceOut], summary="列出序列")
def list_sequences(
    project_id: int | None = None,
    protein_type: str | None = None,
    with_sequence: bool = Query(default=False, description="是否返回完整序列（列表页建议关闭）"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
) -> Page[SequenceOut]:
    query = session.query(ProteinSequence)
    if project_id is not None:
        query = query.filter(ProteinSequence.project_id == project_id)
    if protein_type:
        query = query.filter(ProteinSequence.protein_type == protein_type)
    total = query.count()
    items = (
        query.order_by(ProteinSequence.id.desc()).offset(offset).limit(limit).all()
    )
    return Page[SequenceOut](
        items=[_sequence_out(session, item, with_sequence) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/sequences", response_model=SequenceOut, summary="新增序列")
def create_sequence(payload: SequenceCreateIn, session: Session = Depends(get_db)) -> SequenceOut:
    """序列会先经过校验与清洗，非法字符会被拒绝并给出具体位置。"""
    project = session.get(Project, payload.project_id)
    if project is None:
        raise NotFoundError(f"项目 id={payload.project_id} 不存在")

    check = validate_sequence(payload.sequence)
    if not check.ok:
        raise ValidationError("; ".join(check.errors), detail={"warnings": check.warnings})

    existing = (
        session.query(ProteinSequence)
        .filter(
            ProteinSequence.project_id == payload.project_id,
            ProteinSequence.sha256 == check.sha256,
        )
        .one_or_none()
    )
    if existing is not None:
        return _sequence_out(session, existing)

    record = ProteinSequence(
        project_id=payload.project_id,
        name=payload.name,
        protein_type=payload.protein_type,
        sequence=check.sequence,
        length=check.length,
        sha256=check.sha256,
        note=payload.note,
    )
    session.add(record)
    session.flush()
    return _sequence_out(session, record)


@router.get("/sequences/{sequence_id}", response_model=SequenceOut, summary="序列详情")
def get_sequence(
    sequence_id: int,
    with_sequence: bool = Query(default=True),
    session: Session = Depends(get_db),
) -> SequenceOut:
    record = session.get(ProteinSequence, sequence_id)
    if record is None:
        raise NotFoundError(f"序列 id={sequence_id} 不存在")
    return _sequence_out(session, record, with_sequence)


@router.delete("/sequences/{sequence_id}", response_model=OkOut, summary="删除序列")
def delete_sequence(sequence_id: int, session: Session = Depends(get_db)) -> OkOut:
    record = session.get(ProteinSequence, sequence_id)
    if record is None:
        raise NotFoundError(f"序列 id={sequence_id} 不存在")
    session.delete(record)
    return OkOut(message=f"已删除序列 {record.name}")


@router.get("/dashboard", summary="工作台概览")
def dashboard(
    project_id: int | None = None,
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    """工作台首页所需的汇总数据。"""
    from ...db.models import Job, ModelVersion, MutationCandidate
    from ...jobs.queue import TERMINAL_STATUS

    sequence_query = session.query(func.count(ProteinSequence.id))
    if project_id is not None:
        sequence_query = sequence_query.filter(ProteinSequence.project_id == project_id)

    return {
        "project_count": session.query(func.count(Project.id)).scalar() or 0,
        "sequence_count": sequence_query.scalar() or 0,
        "design_run_count": session.query(func.count(DesignRun.id)).scalar() or 0,
        # 突变候选数（此前误用 count(distinct ProteinSequence.id)，统计的是序列数）
        "mutation_count": session.query(func.count(MutationCandidate.id)).scalar() or 0,
        "unsuccessful_jobs": session.query(func.count(Job.id))
        .filter(Job.status.in_(("failed", "cancelled")))
        .scalar()
        or 0,
        "experiment_count": session.query(func.count(ExperimentRecord.id)).scalar() or 0,
        "model_version_count": session.query(func.count(ModelVersion.id)).scalar() or 0,
        "active_models": session.query(func.count(ModelVersion.id))
        .filter(ModelVersion.is_active.is_(True))
        .scalar()
        or 0,
        "active_jobs": session.query(func.count(Job.id)).filter(Job.status == "running").scalar() or 0,
        "failed_jobs": session.query(func.count(Job.id)).filter(Job.status == "failed").scalar() or 0,
        "protein_type_distribution": dict(
            session.query(ProteinSequence.protein_type, func.count(ProteinSequence.id))
            .group_by(ProteinSequence.protein_type)
            .all()
        ),
        "job_status_distribution": dict(
            session.query(Job.status, func.count(Job.id)).group_by(Job.status).all()
        ),
        "terminal_statuses": list(TERMINAL_STATUS),
    }
