"""FastAPI 应用装配。

启动顺序：日志 -> 目录自检 -> 建库 -> 注册路由 -> 挂载前端静态资源。

真实入口为 ``项目根/run_server.sh``（工作区惯例：项目自带 run_*.sh + conda activate）。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.router import build_api_router
from .core.config import get_settings
from .core.errors import PlatformError
from .core.logging import get_logger, setup_logging
from .db.init_db import init_db

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动 / 关闭钩子。"""
    setup_logging()
    settings = get_settings()
    settings.ensure_runtime_dirs()
    init_db()

    logger.info("=" * 74)
    logger.info("%s v%s 启动中", settings.app_name, settings.version)
    logger.info("  地址      : http://%s:%s", settings.host, settings.port)
    logger.info("  接口文档  : http://%s:%s/docs", settings.host, settings.port)
    logger.info("  数据库    : %s", settings.db_path)
    logger.info("  缓存目录  : %s", settings.cache_dir)
    logger.info("  结构预测  : %s", settings.structure_provider)
    logger.info("  蛋白语言模型: %s (device=%s, fp16=%s)", settings.esm_model, settings.device, settings.use_fp16)
    logger.info("  HF 镜像   : %s", settings.hf_endpoint)
    logger.info("=" * 74)

    try:
        yield
    finally:
        logger.info("服务关闭")

        # 优雅停止后台作业线程池
        try:
            from .jobs.queue import shutdown_queue

            shutdown_queue(wait=False)
        except Exception as exc:  # pragma: no cover
            logger.debug("关闭作业队列时忽略异常: %s", exc)


def create_app() -> FastAPI:
    """构建应用实例。"""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "面向重组蛋白（工业酶、胶原蛋白、蛋白药物）的计算机辅助分子设计平台。\n\n"
            "提供三维结构预测、关键理化性质评估、智能突变设计与实验数据回流迭代能力。"
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------- 统一异常处理 ----------
    @app.exception_handler(PlatformError)
    async def _platform_error_handler(_: Request, exc: PlatformError) -> JSONResponse:
        logger.warning("业务异常 [%s] %s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.code, "message": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "request_validation_error",
                "message": "请求参数校验失败",
                "detail": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("未处理异常: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "服务内部错误，请查看服务端日志",
                "detail": str(exc),
            },
        )

    # ---------- 访问日志与耗时 ----------
    @app.middleware("http")
    async def _access_log(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if request.url.path.startswith("/api"):
            logger.info(
                "%s %s -> %s (%.1f ms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
        return response

    # ---------- 静态资源缓存策略 ----------
    @app.middleware("http")
    async def _static_cache_control(request: Request, call_next):
        """给前端资源补上显式缓存策略，避免升级后用户仍停留在旧界面。

        为什么必须显式声明
        ------------------
        Starlette 的 ``StaticFiles`` **只输出 ETag / Last-Modified，不输出
        Cache-Control**。浏览器在这种情况下会启用"启发式缓存"：把资源在本地
        留存约 ``(当前时间 - Last-Modified) × 10%`` 的时长，且**期间不回源校验**。
        对一个一两天前更新的 ``app.css``，这个窗口长达数小时——服务端换了新样式，
        用户反复刷新仍然看到旧界面，还会误判为"改了没生效"。

        ``no-cache`` 的含义是"使用前必须先回源校验"，并非"不缓存"：资源未变时
        服务端返回 304，开销极小；一旦变化则立即生效。内网单机部署下这是最稳妥的
        取舍——正确性优先，带宽与延迟都不是瓶颈。

        注意：该响应头只能约束**本次之后**的请求。若客户端已经缓存了旧资源，
        需要用户硬刷新（Ctrl+Shift+R）一次才能跳出旧缓存。
        """
        response = await call_next(request)
        path = request.url.path
        if path.startswith(("/static/", "/vendor/")) or path == "/" or path.endswith(".html"):
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

    # ---------- 业务路由（先注册，保证优先于静态资源匹配）----------
    app.include_router(build_api_router())

    # ---------- 前端静态资源 ----------
    static_dir = settings.frontend_dir / "static"
    vendor_dir = settings.frontend_dir / "vendor"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    if vendor_dir.exists():
        app.mount("/vendor", StaticFiles(directory=str(vendor_dir)), name="vendor")

    index_file = settings.frontend_dir / "index.html"
    if index_file.exists():
        # html=True：/ -> index.html，/structure.html -> 同名文件
        app.mount("/", StaticFiles(directory=str(settings.frontend_dir), html=True), name="frontend")
    else:

        @app.get("/", include_in_schema=False)
        def _root_placeholder() -> HTMLResponse:
            return HTMLResponse(
                "<h1>AI 辅助蛋白设计平台</h1>"
                "<p>前端资源尚未就绪，可通过 <a href='/docs'>/docs</a> 使用接口。</p>"
            )

    return app


app = create_app()


def main() -> None:
    """直接运行入口：``python -m backend.app.main``。"""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_config=None,  # 复用项目统一日志配置
    )


if __name__ == "__main__":  # pragma: no cover
    main()
