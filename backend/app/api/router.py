"""API 路由聚合。

约定：健康检查与存活探针**不做鉴权**（供运维与容器编排探活），
其余业务路由在配置了 ``DBZ_API_KEY`` 时统一要求 ``X-API-Key`` 请求头。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from .deps import verify_api_key
from .routes import (
    design,
    experiment,
    health,
    jobs,
    model,
    project,
    property as property_routes,
    sequence,
    structure,
)

#: 无需鉴权的路由（运维修活）
PUBLIC_MODULES = (health,)

#: 需要鉴权的业务路由
SECURED_MODULES = (
    project,
    sequence,
    structure,
    property_routes,
    design,
    experiment,
    model,
    jobs,
)


def build_api_router() -> APIRouter:
    """构建完整的 ``/api`` 路由树。"""
    router = APIRouter(prefix="/api")

    for module in PUBLIC_MODULES:
        router.include_router(module.router)

    secured = APIRouter(dependencies=[Depends(verify_api_key)])
    for module in SECURED_MODULES:
        secured.include_router(module.router)
    router.include_router(secured)

    return router


#: 供 ``main.py`` 直接引用的实例
api_router = build_api_router()

__all__ = ["api_router", "build_api_router"]
