"""ORM 数据模型。

表关系概览::

    Project 1─* ProteinSequence 1─* PropertyRun
                                1─* DesignRun 1─* MutationCandidate
                                1─* ExperimentRecord
    Job      （异步作业，通过 result 回填轻量结果；重结果落各自业务表）
    StructureRecord （按 sequence_sha256 去重的结构缓存）
    ModelVersion    （属性头模型版本仓库）
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Project(Base):
    """项目：序列、设计批次与实验数据的组织单元。"""

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    host_system: Mapped[str] = mapped_column(String(50), default="ecoli")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    sequences: Mapped[list["ProteinSequence"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class ProteinSequence(Base):
    """目标蛋白序列（一条记录对应一次"序列输入"）。"""

    __tablename__ = "sequences"
    __table_args__ = (
        UniqueConstraint("project_id", "sha256", name="uq_sequence_project_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200))
    # collagen | protease | protein_a | generic
    protein_type: Mapped[str] = mapped_column(String(30), default="generic", index=True)
    sequence: Mapped[str] = mapped_column(Text)
    length: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    note: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    project: Mapped[Project] = relationship(back_populates="sequences")
    property_runs: Mapped[list["PropertyRun"]] = relationship(
        back_populates="sequence", cascade="all, delete-orphan"
    )
    design_runs: Mapped[list["DesignRun"]] = relationship(
        back_populates="sequence", cascade="all, delete-orphan"
    )


class StructureRecord(Base):
    """结构预测结果缓存：按 ``sequence_sha256 + provider`` 唯一。"""

    __tablename__ = "structures"
    __table_args__ = (
        UniqueConstraint("sequence_sha256", "provider", name="uq_structure_seq_provider"),
        Index("ix_structures_hash", "sequence_sha256"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sequence_sha256: Mapped[str] = mapped_column(String(64))
    sequence_length: Mapped[int] = mapped_column(Integer)
    provider: Mapped[str] = mapped_column(String(50))
    model_version: Mapped[str | None] = mapped_column(String(100), default=None)
    # PDB 文本落盘路径，避免数据库膨胀
    pdb_path: Mapped[str] = mapped_column(Text)
    plddt: Mapped[list[float]] = mapped_column(JSON, default=list)
    mean_plddt: Mapped[float] = mapped_column(Float, default=0.0)
    truncated: Mapped[bool] = mapped_column(Boolean, default=False)
    segments: Mapped[list[Any]] = mapped_column(JSON, default=list)
    secondary_structure: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    stats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    degradation_reason: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class PropertyRun(Base):
    """一次理化性质预测的完整结果。"""

    __tablename__ = "property_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sequence_id: Mapped[int] = mapped_column(
        ForeignKey("sequences.id", ondelete="CASCADE"), index=True
    )
    job_id: Mapped[str | None] = mapped_column(String(36), default=None, index=True)
    host_system: Mapped[str] = mapped_column(String(50), default="ecoli")
    use_structure: Mapped[bool] = mapped_column(Boolean, default=False)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    results: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    sequence: Mapped[ProteinSequence] = relationship(back_populates="property_runs")


class DesignRun(Base):
    """一次突变设计批次。"""

    __tablename__ = "design_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sequence_id: Mapped[int] = mapped_column(
        ForeignKey("sequences.id", ondelete="CASCADE"), index=True
    )
    job_id: Mapped[str | None] = mapped_column(String(36), default=None, index=True)
    # collagen | protease | protein_a | generic
    category: Mapped[str] = mapped_column(String(30), default="generic", index=True)
    # single | combination | local
    mode: Mapped[str] = mapped_column(String(30), default="single")
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    sequence: Mapped[ProteinSequence] = relationship(back_populates="design_runs")
    candidates: Mapped[list["MutationCandidate"]] = relationship(
        back_populates="design_run",
        cascade="all, delete-orphan",
        order_by="MutationCandidate.rank",
    )


class MutationCandidate(Base):
    """单条突变候选方案（含逐维度可解释评分）。"""

    __tablename__ = "mutations"
    __table_args__ = (Index("ix_mutations_run_rank", "design_run_id", "rank"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    design_run_id: Mapped[int] = mapped_column(
        ForeignKey("design_runs.id", ondelete="CASCADE")
    )
    rank: Mapped[int] = mapped_column(Integer, default=0)
    # ["A123V", "G456P"]
    mutations: Mapped[list[str]] = mapped_column(JSON, default=list)
    positions: Mapped[list[int]] = mapped_column(JSON, default=list)
    total_score: Mapped[float] = mapped_column(Float, default=0.0)
    # [{key,label,raw,weight,contribution,rationale}, ...]
    dimensions: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    flags: Mapped[list[str]] = mapped_column(JSON, default=list)
    rationale: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    design_run: Mapped[DesignRun] = relationship(back_populates="candidates")


class ExperimentRecord(Base):
    """实验实测数据（突变体性质），供模型增量训练与预测对比。"""

    __tablename__ = "experiment_records"
    __table_args__ = (
        Index("ix_experiment_property", "property_name"),
        Index("ix_experiment_sequence", "sequence_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"), default=None, index=True
    )
    sequence_id: Mapped[int | None] = mapped_column(
        ForeignKey("sequences.id", ondelete="SET NULL"), default=None
    )
    design_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("design_runs.id", ondelete="SET NULL"), default=None
    )
    # 突变描述：空字符串表示野生型
    mutation: Mapped[str] = mapped_column(String(200), default="")
    mutated_sequence: Mapped[str | None] = mapped_column(Text, default=None)
    # 属性名，与性质预测键对齐：thermostability / solubility / expression / ...
    property_name: Mapped[str] = mapped_column(String(60))
    measured_value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(30), default=None)
    condition: Mapped[str | None] = mapped_column(String(200), default=None)
    replicate: Mapped[int | None] = mapped_column(Integer, default=None)
    operator: Mapped[str | None] = mapped_column(String(60), default=None)
    measured_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    note: Mapped[str | None] = mapped_column(Text, default=None)
    source_file: Mapped[str | None] = mapped_column(String(300), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ModelVersion(Base):
    """属性头模型版本仓库（模型迭代模块的落地载体）。"""

    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("property_name", "version", name="uq_model_property_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    property_name: Mapped[str] = mapped_column(String(60), index=True)
    version: Mapped[str] = mapped_column(String(40))
    algo: Mapped[str] = mapped_column(String(60), default="ridge")
    base_model: Mapped[str] = mapped_column(String(120), default="")
    n_samples: Mapped[int] = mapped_column(Integer, default=0)
    n_features: Mapped[int] = mapped_column(Integer, default=0)
    # {"r2":..,"spearman":..,"mae":..,"cv_r2":..}
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    artifact_path: Mapped[str | None] = mapped_column(Text, default=None)
    status: Mapped[str] = mapped_column(String(20), default="ready")  # ready|training|failed
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    #: 本次训练已消费的最大 ExperimentRecord.id，作为增量训练的水位线。
    #: **不能**用 created_at 做水位线：SQLite 的 CURRENT_TIMESTAMP 只到秒，
    #: 同一秒内写入的记录会被判定为"没有新增"，导致增量训练永远无法触发。
    last_record_id: Mapped[int | None] = mapped_column(Integer, default=None)
    note: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Job(Base):
    """异步作业记录（结构预测 / 性质预测 / 突变设计 / 增量训练）。"""

    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_status_created", "status", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"), default=None, index=True
    )
    # structure | property | design | train
    kind: Mapped[str] = mapped_column(String(30), index=True)
    # pending | running | success | failed | cancelled
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str | None] = mapped_column(String(120), default=None)
    title: Mapped[str | None] = mapped_column(String(200), default=None)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
