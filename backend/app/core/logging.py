"""统一日志。

约定：
* 作业生命周期（submit / start / done / fail）与外部调用耗时统一走 ``get_logger``。
* **不记录完整序列**，长序列仅记录长度与 sha256 前 12 位，避免日志泄漏合作方数据。
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

from .config import get_settings

_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-34s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int | str | None = None, to_file: bool = True) -> None:
    """配置根 logger。重复调用安全（幂等）。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    resolved = level if level is not None else (logging.DEBUG if settings.debug else logging.INFO)
    if isinstance(resolved, str):
        resolved = logging.getLevelName(resolved.upper())
        if not isinstance(resolved, int):
            resolved = logging.INFO

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if to_file:
        log_dir = Path(settings.logs_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "platform.log", encoding="utf-8")
        handlers.append(file_handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(resolved)
    for handler in handlers:
        root.addHandler(handler)

    # 第三方库降噪
    for noisy in ("httpx", "httpcore", "urllib3", "transformers", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """获取 logger；首次调用时自动完成配置。"""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)


def sequence_fingerprint(sequence: str) -> str:
    """序列指纹：用于日志与缓存键，不暴露原始序列内容。"""
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:12]


def describe_sequence(sequence: str) -> str:
    """日志安全地描述一条序列。"""
    return f"len={len(sequence)} fp={sequence_fingerprint(sequence)}"
