"""数据库引擎与会话管理。

选型 SQLite + WAL：单机企业内网部署，无运维成本；WAL 模式允许"读写并发"，
满足"前端轮询作业进度"与"后台作业写结果"同时进行的场景。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ..core.config import get_settings


class Base(DeclarativeBase):
    """ORM 声明基类。"""


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _configure_sqlite(dbapi_connection, _connection_record) -> None:
    """每条新连接都应用 WAL 与外键约束。"""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=15000")
    finally:
        cursor.close()


def get_engine() -> Engine:
    """全局引擎单例。"""
    global _engine
    if _engine is None:
        settings = get_settings()
        url = f"sqlite:///{settings.db_path}"
        _engine = create_engine(
            url,
            echo=False,
            future=True,
            # 后台作业线程与请求线程共用引擎，需放开线程检查
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        event.listen(_engine, "connect", _configure_sqlite)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """会话工厂单例。"""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(), autoflush=False, autocommit=False, expire_on_commit=False
        )
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """事务作用域：正常提交，异常回滚。"""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """测试用：释放引擎与会话工厂。"""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
