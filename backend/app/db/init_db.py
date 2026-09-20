"""数据库初始化（幂等）。"""

from __future__ import annotations

from ..core.config import get_settings
from ..core.logging import get_logger
from .base import Base, get_engine, session_scope
from .models import Project

logger = get_logger(__name__)

DEFAULT_PROJECT_NAME = "默认项目"


def _sync_schema() -> list[str]:
    """轻量 schema 迁移：为已存在的表补齐新增列。

    ``create_all`` 只建缺失的表，**不会**给已有表加列。企业现场升级版本时
    数据库里已有数据，不能简单删库重建，因此这里显式检查并 ``ALTER TABLE``。
    只处理"新增可空列"这一种最安全的变更；涉及改类型/加约束的变更仍需人工处理，
    此时会在日志中给出明确告警。
    """
    from sqlalchemy import inspect

    engine = get_engine()
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    applied: list[str] = []

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        present = {column["name"] for column in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present or column.primary_key:
                continue
            if not column.nullable and column.default is None and column.server_default is None:
                logger.warning(
                    "表 %s 缺少非空列 %s 且无默认值，无法自动迁移，请人工处理",
                    table.name,
                    column.name,
                )
                continue
            column_type = column.type.compile(engine.dialect)
            statement = f'ALTER TABLE {table.name} ADD COLUMN {column.name} {column_type}'
            try:
                with engine.begin() as connection:
                    connection.exec_driver_sql(statement)
                applied.append(f"{table.name}.{column.name}")
                logger.info("schema 迁移: %s", applied[-1])
            except Exception as exc:  # pragma: no cover - 迁移失败需人工介入
                logger.error("schema 迁移失败 %s.%s: %s", table.name, column.name, exc)

    return applied


def init_db(create_default_project: bool = True) -> None:
    """建表 + 补齐列 + 补齐默认项目。可重复调用。"""
    settings = get_settings()
    settings.ensure_runtime_dirs()

    engine = get_engine()
    Base.metadata.create_all(engine)
    migrated = _sync_schema()
    logger.info(
        "数据库已就绪: %s%s",
        settings.db_path,
        f"（迁移了 {len(migrated)} 个新增列）" if migrated else "",
    )

    if create_default_project:
        ensure_default_project()


def ensure_default_project() -> int:
    """确保存在默认项目，返回其 id。"""
    with session_scope() as session:
        existing = (
            session.query(Project).filter(Project.name == DEFAULT_PROJECT_NAME).one_or_none()
        )
        if existing is not None:
            return existing.id
        project = Project(
            name=DEFAULT_PROJECT_NAME,
            description="平台初始化自动创建；可直接使用，也可新建独立项目。",
            host_system="ecoli",
        )
        session.add(project)
        session.flush()
        logger.info("已创建默认项目 id=%s", project.id)
        return project.id


if __name__ == "__main__":  # pragma: no cover - 手工初始化入口
    init_db()
