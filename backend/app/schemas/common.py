"""通用响应模型：分页、作业状态。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    """分页响应。"""

    items: list[T]
    total: int
    limit: int
    offset: int


class OkOut(BaseModel):
    """简单成功响应。"""

    ok: bool = True
    message: str = ""
    data: dict[str, Any] | None = None


class JobOut(BaseModel):
    """作业状态与结果。"""

    id: str
    kind: str
    status: str = Field(description="pending | running | success | failed | cancelled")
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    stage: str | None = None
    title: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in ("success", "failed", "cancelled")


class JobAccepted(BaseModel):
    """作业已受理。"""

    job_id: str
    status: str = "pending"
    message: str = "作业已提交，请轮询 /api/jobs/{job_id} 获取进度"
