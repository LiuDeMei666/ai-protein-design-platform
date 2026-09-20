"""FastAPI 依赖注入。"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, Header
from sqlalchemy.orm import Session

from ..core.config import Settings, get_settings
from ..core.errors import AuthError
from ..db.base import get_session_factory


def get_db() -> Iterator[Session]:
    """请求级数据库会话。

    **必须在请求正常结束时 commit**。SQLAlchemy 的 ``Session.close()`` 会回滚
    未提交的更改，因此如果只 ``flush`` 不 ``commit``，所有直接改库的路由
    （新增/删除序列与项目、录入实验记录、激活模型版本等）都会静默丢数据——
    接口返回 200，但数据没落库。这个问题由接口层测试用 "创建后再查询" 的
    序列模式抓出来，属于必须修的缺陷。
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_config() -> Settings:
    """全局配置。"""
    return get_settings()


def resolve_project_id(session: Session, project_id: int | None) -> int | None:
    """解析项目 id：显式给出则用之，未给出则回落到默认项目。

    为什么要回落：作业若不带 project_id，领域结果（结构/性质/设计批次）就**不会落库**，
    前端的"导出候选方案""历史记录"等接口会拿不到任何数据。默认项目是平台初始化
    时创建的，直接用它可保证首次使用就有完整闭环。
    """
    if project_id is not None:
        return project_id
    from ..db.init_db import DEFAULT_PROJECT_NAME
    from ..db.models import Project

    project = session.query(Project).filter(Project.name == DEFAULT_PROJECT_NAME).one_or_none()
    return project.id if project is not None else None


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_config),
) -> None:
    """可选鉴权：``DBZ_API_KEY`` 未配置时放行，便于内网快速试用。"""
    if not settings.require_api_key:
        return
    if x_api_key != settings.api_key:
        raise AuthError("API Key 校验失败", detail={"hint": "请在请求头携带 X-API-Key"})
