"""性质预测相关的请求 / 响应模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceOut(BaseModel):
    """一条可解释性证据。"""

    label: str
    value: float | str
    contribution: float = Field(description="折算后对总分的实际贡献；各项之和等于 total score")
    rationale: str = ""


class MetricOut(BaseModel):
    """单个指标结果。"""

    key: str
    label: str
    score: float = Field(description="0-100，越高越好")
    risk: str = Field(description="low | medium | high")
    risk_label: str = ""
    unit: str = ""
    value: float | None = None
    algorithm: str = Field(default="", description="算法来源，便于在报告中标注依据")
    rationale: str = ""
    confidence: float = Field(default=0.5, description="0-1，本项结论的可信度")
    evidence: list[EvidenceOut] = Field(default_factory=list)
    locus: list[dict[str, Any]] = Field(
        default_factory=list, description="位点级明细，供位点轨道图渲染"
    )
    meta: dict[str, Any] = Field(default_factory=dict)


class BiophysOut(BaseModel):
    """基础生物物理量（确定性计算，无模型）。"""

    model_config = ConfigDict(extra="allow")

    length: int = 0
    molecular_weight_da: float = 0.0
    molecular_weight_kda: float = 0.0
    isoelectric_point: float = 0.0
    gravy: float = 0.0
    aromaticity: float = 0.0
    instability_index: float = 0.0
    instability_class: str = ""
    net_charge_ph7: float = 0.0
    counts: dict[str, int] = Field(default_factory=dict)
    ratios: dict[str, float] = Field(default_factory=dict)
    group_ratios: dict[str, float] = Field(default_factory=dict)
    special_residues: dict[str, int] = Field(default_factory=dict)
    secondary_structure_fraction_chou_fasman: dict[str, float] = Field(default_factory=dict)
    charge_profile: dict[str, list[float]] = Field(default_factory=dict)
    n_end_rule: dict[str, str] = Field(default_factory=dict)
    algorithm: dict[str, str] = Field(default_factory=dict)


class RadarItem(BaseModel):
    """雷达图一项。"""

    key: str
    label: str
    score: float


class SummaryOut(BaseModel):
    """性质汇总。"""

    radar: list[RadarItem] = Field(default_factory=list)
    risk_counts: dict[str, int] = Field(default_factory=dict)
    overall_score: float = 0.0
    weakest_metrics: list[dict[str, Any]] = Field(default_factory=list)
    protein_type: str = "generic"
    host_system: str = "ecoli"
    esm_used: bool = False
    structure_used: bool = False


class PropertyPredictIn(BaseModel):
    """性质预测请求。"""

    sequence: str = Field(description="氨基酸序列（可含 FASTA 头）")
    protein_type: Literal["collagen", "protease", "protein_a", "generic"] = "generic"
    host_system: str = Field(default="ecoli", description="宿主表达体系，决定密码子偏好参考表")
    use_structure: bool = Field(
        default=True, description="是否先预测结构并用结构统计增强性质评估"
    )
    provider: str | None = Field(default=None, description="结构预测 Provider，留空自动选择")
    use_esm: bool = Field(
        default=True, description="是否使用 ESM-2 计算序列自然度（需要 GPU 与已下载权重）"
    )
    dna_sequence: str | None = Field(
        default=None, description="可选：基因序列，提供时将计算真实 CAI"
    )
    project_id: int | None = Field(default=None, description="归属项目；留空则使用默认项目")
    name: str | None = Field(default=None, description="序列名称（落库时使用）")


class PropertyOut(BaseModel):
    """性质预测结果。"""

    length: int
    protein_type: str
    host_system: str
    biophys: BiophysOut | None = None
    metrics: dict[str, MetricOut] = Field(default_factory=dict)
    summary: SummaryOut | None = None
    warnings: list[str] = Field(default_factory=list)
    esm_used: bool = False
    structure_used: bool = False


class PropertyTrackOut(BaseModel):
    """位点轨道数据（供前端横向轨道图渲染）。"""

    length: int
    aggregation: list[dict[str, Any]] = Field(default_factory=list)
    loci: list[dict[str, Any]] = Field(default_factory=list)
    legend: dict[str, str] = Field(default_factory=dict)
