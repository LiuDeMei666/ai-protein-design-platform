"""作业管理路由。"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ...core.errors import NotFoundError
from ...db.models import Job
from ...jobs.queue import TERMINAL_STATUS, cancel_job, queue_status
from ...schemas.common import JobOut, OkOut, Page
from ..deps import get_db

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _to_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        kind=job.kind,
        status=job.status,
        progress=float(job.progress or 0.0),
        stage=job.stage,
        title=job.title,
        params=job.params or {},
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.get("", response_model=Page[JobOut], summary="列出作业")
def list_jobs(
    status: Literal["pending", "running", "success", "failed", "cancelled"] | None = None,
    kind: str | None = None,
    project_id: int | None = None,
    active_only: bool = Query(default=False, description="只看未完成的作业"),
    limit: int = Query(default=30, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
) -> Page[JobOut]:
    """按状态/类型/项目筛选作业，默认按创建时间倒序。"""
    query = session.query(Job)
    if status:
        query = query.filter(Job.status == status)
    if kind:
        query = query.filter(Job.kind == kind)
    if project_id is not None:
        query = query.filter(Job.project_id == project_id)
    if active_only:
        query = query.filter(Job.status.notin_(TERMINAL_STATUS))

    total = query.count()
    items = query.order_by(Job.created_at.desc()).offset(offset).limit(limit).all()
    return Page[JobOut](
        items=[_to_out(job) for job in items], total=total, limit=limit, offset=offset
    )


@router.get("/queue", summary="队列运行状态")
def queue_state() -> dict[str, Any]:
    """线程池与重计算信号量的实时状态。"""
    return queue_status()


@router.get("/{job_id}", response_model=JobOut, summary="查询作业状态与结果")
def get_job(job_id: str, session: Session = Depends(get_db)) -> JobOut:
    """前端以 1-2 秒间隔轮询此接口获取进度。"""
    job = session.get(Job, job_id)
    if job is None:
        raise NotFoundError(f"作业 {job_id} 不存在")
    return _to_out(job)


@router.post("/{job_id}/cancel", response_model=OkOut, summary="取消作业")
def cancel(job_id: str, session: Session = Depends(get_db)) -> OkOut:
    """请求取消。排队中的立即取消；运行中的会在下一个进度检查点退出。"""
    job = session.get(Job, job_id)
    if job is None:
        raise NotFoundError(f"作业 {job_id} 不存在")
    if job.status in TERMINAL_STATUS:
        return OkOut(ok=False, message=f"作业已处于终态 {job.status}，无需取消")
    ok = cancel_job(job_id)
    return OkOut(
        ok=ok,
        message="取消请求已提交" if ok else "取消失败",
        data={"job_id": job_id},
    )


@router.delete("/{job_id}", response_model=OkOut, summary="删除作业记录")
def delete_job(job_id: str, session: Session = Depends(get_db)) -> OkOut:
    """仅允许删除终态作业记录。"""
    job = session.get(Job, job_id)
    if job is None:
        raise NotFoundError(f"作业 {job_id} 不存在")
    if job.status not in TERMINAL_STATUS:
        return OkOut(ok=False, message="作业尚未结束，不能删除；请先取消")
    session.delete(job)
    return OkOut(message="已删除", data={"job_id": job_id})
