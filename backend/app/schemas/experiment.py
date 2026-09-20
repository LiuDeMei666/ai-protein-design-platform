"""实验数据与模型迭代相关的请求 / 响应模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExperimentRecordIn(BaseModel):
    """单条实验记录（录入用）。"""

    mutation: str = Field(default="", description="突变标签，如 A123V；野生型留空")
    property_name: str = Field(description="属性名，需与性质预测的指标键对齐")
    measured_value: float
    unit: str | None = None
    condition: str | None = Field(default=None, description="测定条件，如 pH 7.0, 25 °C")
    replicate: int | None = None
    operator: str | None = None
    sequence_id: int | None = None
    design_run_id: int | None = None
    mutated_sequence: str | None = None
    note: str | None = None


class ExperimentRecordOut(BaseModel):
    """实验记录（输出）。

    必须开启 ``from_attributes``：Pydantic v2 不再默认支持从 ORM 对象取值，
    缺少该配置时 ``model_validate(orm_obj)`` 会直接抛 ValidationError（实测踩过，
    表现为实验记录列表接口 500）。
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int | None = None
    sequence_id: int | None = None
    design_run_id: int | None = None
    mutation: str = ""
    property_name: str
    measured_value: float
    unit: str | None = None
    condition: str | None = None
    replicate: int | None = None
    operator: str | None = None
    measured_at: datetime | None = None
    note: str | None = None
    source_file: str | None = None
    created_at: datetime | None = None


class IngestErrorOut(BaseModel):
    """一行数据的校验错误。"""

    row: int = Field(description="原始文件中的行号（1-based，含表头）")
    field: str = ""
    message: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


class IngestReportOut(BaseModel):
    """批量导入报告。"""

    total_rows: int = 0
    accepted_rows: int = 0
    rejected_rows: int = 0
    duplicate_rows: int = 0
    detected_columns: dict[str, str] = Field(default_factory=dict)
    unmapped_columns: list[str] = Field(default_factory=list)
    property_counts: dict[str, int] = Field(default_factory=dict)
    errors: list[IngestErrorOut] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    dry_run: bool = False


class IngestResultOut(BaseModel):
    """导入结果。"""

    report: IngestReportOut
    inserted_ids: list[int] = Field(default_factory=list)
    preview: list[dict[str, Any]] = Field(default_factory=list)


class CalibrationOut(BaseModel):
    """一阶线性校准系数：``实测 ≈ slope × 平台评分 + intercept``。

    平台指标是 0-100 评分，实测值有各自量纲（°C、%、U/mg），两者不同尺度，
    所以要先做一次线性校准，MAE / RMSE / R² 才有物理意义。
    """

    slope: float
    intercept: float


class PropertyComparisonOut(BaseModel):
    """单个属性的预测-实测对比。"""

    property_name: str
    label: str = ""
    n_pairs: int = 0
    pearson_r: float | None = None
    spearman_rho: float | None = None
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    bias: float | None = None
    unit: str | None = None
    verdict: str = ""
    points: list[dict[str, Any]] = Field(default_factory=list)
    worst_offsets: list[dict[str, Any]] = Field(default_factory=list)
    #: 校准系数。**必须声明**：响应模型只保留声明过的字段，
    #: 漏声明会让 services/experiment/compare.py 里算好的 slope/intercept
    #: 在序列化时被静默丢弃，前端读 item.calibration.slope 直接抛 TypeError。
    calibration: CalibrationOut | None = None


class ComparisonOut(BaseModel):
    """预测-实测对比总报告。"""

    sequence_id: int | None = None
    design_run_id: int | None = None
    total_records: int = 0
    matched_pairs: int = 0
    unmatched_records: int = 0
    properties: list[PropertyComparisonOut] = Field(default_factory=list)
    overall_verdict: str = ""
    recommendations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ModelVersionOut(BaseModel):
    """模型版本。"""

    id: int
    property_name: str
    version: str
    algo: str = ""
    base_model: str = ""
    n_samples: int = 0
    n_features: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    status: str = "ready"
    is_active: bool = False
    note: str | None = None
    created_at: datetime | None = None


class TrainRequestIn(BaseModel):
    """增量训练请求。"""

    property_name: str = Field(description="目标属性（对应实验记录的 property_name）")
    algo: str = Field(default="ridge", description="ridge | hgb")
    min_samples: int = Field(default=8, ge=2, description="低于该样本数则拒绝训练")
    test_ratio: float = Field(default=0.25, ge=0.0, le=0.5)
    cv_folds: int = Field(default=5, ge=2, le=10)
    notes: str | None = None


class TrainResultOut(BaseModel):
    """增量训练结果。"""

    version: ModelVersionOut
    metrics: dict[str, Any] = Field(default_factory=dict)
    feature_dim: int = 0
    train_samples: int = 0
    test_samples: int = 0
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class TemplateColumnOut(BaseModel):
    """录入模板的列定义。"""

    name: str
    label: str
    required: bool = False
    example: str = ""
    description: str = ""
