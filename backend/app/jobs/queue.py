"""进程内异步作业队列。

为什么不用 Celery/Redis
-----------------------
企业内网单机部署，引入消息中间件会显著提高运维成本。结构预测与突变扫描属于
分钟级任务，用一个进程内线程池 + 数据库作业表即可满足"提交-轮询-取结果"的
全部需求，且**服务重启后历史作业仍可查**（状态存在数据库里）。

三个关键设计
------------
1. **状态以数据库为准**：每次状态变化都写库，前端轮询读库，不存在"内存状态与
   界面不一致"的问题。
2. **重计算作业串行化**：结构预测与 ESM-2 都要抢 V100 显存，用信号量保证同一
   时刻只有一个重计算作业在跑（可通过 ``DBZ_SERIALIZE_HEAVY_JOBS=false`` 关闭）。
3. **取消是协作式的**：作业在执行到进度检查点时读取数据库中的 cancel 标记，
   因此"取消"不是立刻生效，但**不会残留半个结果**（取消后结果不落库）。
"""

from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from ..core.config import get_settings
from ..core.errors import PlatformError
from ..core.logging import get_logger
from ..db.base import session_scope
from ..db.models import Job

logger = get_logger(__name__)

JobHandler = Callable[[dict[str, Any], "JobContext"], dict[str, Any]]

_executor: ThreadPoolExecutor | None = None
_heavy_semaphore: threading.Semaphore | None = None
_lock = threading.RLock()
_futures: dict[str, Future] = {}

TERMINAL_STATUS = ("success", "failed", "cancelled")


@dataclass
class JobContext:
    """作业执行上下文。"""

    job_id: str
    kind: str
    _cancelled: bool = False
    _last_stage: str | None = None
    _last_progress: float = 0.0
    logs: list[str] = field(default_factory=list)

    def progress(self, value: float, stage: str | None = None) -> None:
        """上报进度（0-1）。同时检查取消标记。"""
        value = max(0.0, min(1.0, float(value)))
        stage_changed = stage is not None and stage != self._last_stage
        # 避免高频写库：进度变化 <2% 且阶段未变时跳过
        if not stage_changed and abs(value - self._last_progress) < 0.02 and value < 1.0:
            return
        self._last_progress = value
        self._last_stage = stage or self._last_stage
        _update_job(self.job_id, progress=value, stage=stage)
        if self.is_cancelled:
            raise JobCancelled(f"作业 {self.job_id} 已被取消")

    def log(self, message: str) -> None:
        """记录作业内日志（写入内存，失败时随错误一并落库）。"""
        self.logs.append(message)
        logger.info("[job %s] %s", self.job_id[:8], message)

    @property
    def is_cancelled(self) -> bool:
        if self._cancelled:
            return True
        self._cancelled = _is_cancelled(self.job_id)
        return self._cancelled


class JobCancelled(Exception):
    """作业被用户取消。"""


# --------------------------------------------------------------------------- #
# 数据库状态读写
# --------------------------------------------------------------------------- #
def _update_job(
    job_id: str,
    *,
    status: str | None = None,
    progress: float | None = None,
    stage: str | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    started: bool = False,
    finished: bool = False,
) -> None:
    """更新作业状态（独立短事务，避免长时间持锁）。"""
    try:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                return
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = progress
            if stage is not None:
                job.stage = stage
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            if started:
                job.started_at = datetime.now()
            if finished:
                job.finished_at = datetime.now()
    except Exception as exc:  # pragma: no cover - 状态写入失败不应影响作业本身
        logger.warning("更新作业状态失败 %s: %s", job_id, exc)


def _is_cancelled(job_id: str) -> bool:
    try:
        with session_scope() as session:
            job = session.get(Job, job_id)
            return job is not None and job.status == "cancelled"
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# 队列
# --------------------------------------------------------------------------- #
def get_executor() -> ThreadPoolExecutor:
    """线程池单例。"""
    global _executor
    if _executor is None:
        with _lock:
            if _executor is None:
                settings = get_settings()
                workers = max(1, settings.job_workers)
                _executor = ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="dbz-job"
                )
                logger.info("作业线程池已创建：workers=%d", workers)
    return _executor


def get_heavy_semaphore() -> threading.Semaphore | None:
    """重计算作业信号量（结构预测 / ESM-2 互斥）。"""
    global _heavy_semaphore
    settings = get_settings()
    if not settings.serialize_heavy_jobs:
        return None
    if _heavy_semaphore is None:
        with _lock:
            if _heavy_semaphore is None:
                _heavy_semaphore = threading.Semaphore(1)
    return _heavy_semaphore


def submit_job(
    kind: str,
    handler: JobHandler,
    params: dict[str, Any] | None = None,
    *,
    project_id: int | None = None,
    title: str | None = None,
    heavy: bool = False,
) -> str:
    """提交一个作业，返回 job_id。"""
    job_id = uuid.uuid4().hex
    payload = params or {}

    with session_scope() as session:
        session.add(
            Job(
                id=job_id,
                project_id=project_id,
                kind=kind,
                status="pending",
                progress=0.0,
                stage="已入队",
                title=title,
                params=_sanitize_params(payload),
            )
        )

    logger.info("作业已提交 id=%s kind=%s title=%s", job_id, kind, title)

    future = get_executor().submit(_run_job, job_id, kind, handler, payload, heavy)
    with _lock:
        _futures[job_id] = future
    return job_id


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    """作业参数落库前脱敏：不保存完整序列，只留长度与指纹。"""
    from ..core.logging import sequence_fingerprint

    safe: dict[str, Any] = {}
    for key, value in params.items():
        if key in ("sequence", "dna_sequence") and isinstance(value, str):
            safe[f"{key}_length"] = len(value)
            safe[f"{key}_fingerprint"] = sequence_fingerprint(value)
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, (list, tuple)):
            safe[key] = list(value)[:50]
        elif isinstance(value, dict):
            safe[key] = {str(k): str(v)[:100] for k, v in list(value.items())[:30]}
        else:
            safe[key] = str(value)[:200]
    return safe


def _run_job(
    job_id: str, kind: str, handler: JobHandler, params: dict[str, Any], heavy: bool
) -> None:
    """作业执行包装：负责状态流转、异常兜底与信号量。"""
    context = JobContext(job_id=job_id, kind=kind)
    _update_job(job_id, status="running", started=True, progress=0.0, stage="开始执行")

    semaphore = get_heavy_semaphore() if heavy else None
    acquired = False
    try:
        if semaphore is not None:
            context.log("等待重计算资源（GPU 串行化）…")
            semaphore.acquire()
            acquired = True

        if context.is_cancelled:
            raise JobCancelled("提交后立即被取消")

        result = handler(params, context)
        _update_job(
            job_id,
            status="success",
            progress=1.0,
            stage="完成",
            result=result,
            finished=True,
        )
        context.log("作业完成")

    except JobCancelled:
        _update_job(
            job_id,
            status="cancelled",
            stage="已取消",
            error="作业被用户取消（结果未保存）",
            finished=True,
        )
        logger.info("作业已取消 id=%s", job_id)

    except PlatformError as exc:
        _update_job(
            job_id,
            status="failed",
            stage="失败",
            error=f"{exc.message}" + (f" | 详情: {exc.detail}" if exc.detail else ""),
            finished=True,
        )
        logger.warning("作业业务失败 id=%s: %s", job_id, exc.message)

    except Exception as exc:
        detail = traceback.format_exc()
        _update_job(
            job_id,
            status="failed",
            stage="失败",
            error=f"未预期错误: {exc}\n{detail[-1500:]}",
            finished=True,
        )
        logger.exception("作业异常 id=%s", job_id)

    finally:
        if acquired and semaphore is not None:
            semaphore.release()
        with _lock:
            _futures.pop(job_id, None)


def cancel_job(job_id: str) -> bool:
    """请求取消作业。

    若作业还在排队（pending），直接置为 cancelled；若正在运行，则设置标记，
    由作业在下一个进度检查点自行退出。
    """
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return False
        if job.status in TERMINAL_STATUS:
            return False
        if job.status == "pending":
            job.status = "cancelled"
            job.stage = "已取消（未开始执行）"
            job.error = "作业在排队阶段被取消"
            job.finished_at = datetime.now()
        else:
            # 运行中的作业需要标记，由 JobContext.is_cancelled 轮询到
            job.stage = "取消中…"
            job.status = "cancelled"
            job.finished_at = datetime.now()
    logger.info("已请求取消作业 id=%s", job_id)
    return True


def shutdown_queue(wait: bool = False) -> None:
    """关闭线程池。"""
    global _executor, _heavy_semaphore
    with _lock:
        if _executor is not None:
            _executor.shutdown(wait=wait, cancel_futures=not wait)
            _executor = None
            logger.info("作业线程池已关闭 (wait=%s)", wait)
        _heavy_semaphore = None
        _futures.clear()


def queue_status() -> dict[str, Any]:
    """队列运行态（供健康检查与状态栏展示）。"""
    executor = _executor
    semaphore = _heavy_semaphore
    return {
        "workers": getattr(executor, "_max_workers", 0) if executor else 0,
        "active_jobs": len(_futures),
        "serialize_heavy_jobs": get_settings().serialize_heavy_jobs,
        "heavy_slot_free": (semaphore is None) or (semaphore._value > 0),  # noqa: SLF001
    }
