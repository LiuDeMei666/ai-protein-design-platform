"""突变设计相关的请求 / 响应模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ProteinType = Literal["collagen", "protease", "protein_a", "generic"]
DesignMode = Literal["single", "combination", "local", "manual"]


class ScoreDimensionOut(BaseModel):
    """一个评分维度。"""

    key: str
    label: str
    raw: float = Field(description="该维度得分（0-100，越高越好）")
    weight: float
    contribution: float = Field(description="对总分的实际贡献；所有维度之和 = total_score")
    rationale: str


class MutationCandidateOut(BaseModel):
    """一条突变候选方案。"""

    mutations: list[str] = Field(description='突变标签，如 ["A123V"]')
    positions: list[int] = Field(description="1-based 残基位置（前端直接使用）")
    site_count: int = 1
    total_score: float
    dimensions: list[ScoreDimensionOut] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list, description="风险告警")
    category: str = "generic"
    rationale: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class ExclusionOut(BaseModel):
    """被排除的位点。"""

    position: int
    residue: str
    reason: str


class ScanPlanOut(BaseModel):
    """扫描计划明细（让用户清楚"为什么某些位点没被推荐"）。"""

    position_count: int
    candidate_count: int
    positions: list[int] = Field(default_factory=list)
    exclusions: list[ExclusionOut] = Field(default_factory=list)
    protected: dict[str, str] = Field(default_factory=dict)
    downsampled: bool = False
    downsample_note: str = ""
    notes: list[str] = Field(default_factory=list)


class CombinationOut(BaseModel):
    """组合突变候选。"""

    mutations: list[str]
    positions: list[int]
    estimated_score: float
    refined_score: float | None = None
    final_score: float
    epitasis_penalty: float = 0.0
    refined: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class DesignRequestIn(BaseModel):
    """突变设计请求。"""

    sequence: str = Field(description="氨基酸序列（可含 FASTA 头）")
    protein_type: ProteinType = "generic"
    mode: DesignMode = "single"
    host_system: str = "ecoli"
    region: list[int] | None = Field(
        default=None, description="限定突变区段 [start, end]，1-based 闭区间；留空为全长"
    )
    max_positions: int | None = Field(default=None, description="参与扫描的位点上限")
    top_n: int = Field(default=50, ge=1, le=2000, description="返回的候选数量")
    max_sites: int = Field(default=3, ge=1, le=5, description="组合突变最多叠加位点数")
    beam_width: int = Field(default=8, ge=1, le=32, description="组合搜索束宽")
    use_structure: bool = Field(default=True, description="是否使用结构信息（pLDDT/SASA/二级结构）")
    provider: str | None = Field(default=None, description="结构预测 Provider，留空自动选择")
    project_id: int | None = Field(default=None, description="归属项目；留空则使用默认项目")
    name: str | None = Field(default=None, description="序列名称（落库时使用）")

    # ---------- 人工模式（mode="manual"）专用 ----------
    #: 人工选定的目标位点（1-based）。自动模式忽略。
    target_positions: list[int] | None = Field(
        default=None, description="人工选定的目标位点（1-based），仅 manual 模式使用"
    )
    substitutions: dict[str, list[str]] | None = Field(
        default=None,
        description='每位点的候选氨基酸，如 {"23": ["A", "S"]}；缺省时按残基类别套用默认集合',
    )
    locked_positions: list[int] = Field(
        default_factory=list,
        description="人工声明的禁止突变位点（1-based，如二硫键 Cys）；平台不做自动识别",
    )
    mutation_notes: dict[str, str] | None = Field(
        default=None,
        description="突变备注，键为突变标签（如 W23A）或位点；导出 CSV 时写入 mutation_note 列",
    )

    @field_validator("region")
    @classmethod
    def _check_region(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError("region 必须是 [start, end] 两个整数")
        if value[0] >= value[1]:
            raise ValueError("region 的 start 必须小于 end")
        return value


class DesignSummaryOut(BaseModel):
    """设计结果汇总。"""

    candidate_count: int = 0
    combination_count: int = 0
    score_max: float = 0.0
    score_mean: float = 0.0
    score_median: float = 0.0
    flagged_count: int = 0
    evaluated_candidates: int = 0
    elapsed_seconds: float = 0.0
    top_candidate: dict[str, Any] = Field(default_factory=dict)
    hotspots: list[dict[str, Any]] = Field(default_factory=list)


class DesignResultOut(BaseModel):
    """突变设计结果。"""

    length: int
    protein_type: str
    mode: str
    plan: ScanPlanOut | None = None
    rulepack: dict[str, Any] = Field(default_factory=dict)
    candidates: list[MutationCandidateOut] = Field(default_factory=list)
    combinations: list[CombinationOut] = Field(default_factory=list)
    summary: DesignSummaryOut | None = None
    hints: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    zero_shot_stats: dict[str, Any] = Field(default_factory=dict)
    manual: dict[str, Any] = Field(
        default_factory=dict, description="人工模式专有信息；自动模式为空对象"
    )


class DesignCatalogOut(BaseModel):
    """设计能力元信息。"""

    modes: list[dict[str, Any]] = Field(default_factory=list)
    protein_types: list[dict[str, Any]] = Field(default_factory=list)
    scoring_dimensions: dict[str, Any] = Field(default_factory=dict)
    combination: dict[str, Any] = Field(default_factory=dict)
    scan: dict[str, Any] = Field(default_factory=dict)
    manual: dict[str, Any] = Field(default_factory=dict, description="人工模式默认值与说明")


# --------------------------------------------------------------------------- #
# 人工指定突变（MVP 验证工作流）
# --------------------------------------------------------------------------- #


class ManualPlanRequestIn(BaseModel):
    """人工突变清单的预览 / 导出请求。

    只做规划与校验，**不触发任何评估**（不调用 ESM-2、不查结构），
    因此可以同步返回，供前端在真正提交作业之前先把清单摆给用户确认。
    """

    sequence: str = Field(description="野生型序列（可含 FASTA 头）")
    name: str | None = Field(default=None, description="序列名，用于生成导出的 sequence_id")
    protein_type: ProteinType = "generic"
    target_positions: list[int] = Field(
        min_length=1, description="人工选定的目标位点（1-based）"
    )
    substitutions: dict[str, list[str]] | None = Field(
        default=None,
        description='每位点的候选氨基酸，如 {"23": ["A", "S"]}；缺省时套用默认集合',
    )
    locked_positions: list[int] = Field(
        default_factory=list, description="禁止突变位点（1-based），如二硫键 Cys"
    )
    mutation_notes: dict[str, str] | None = Field(
        default=None, description="突变备注，键为突变标签或位点"
    )
    include_wild_type: bool = Field(
        default=True, description="导出时是否包含野生型基准对照（需求要求保留）"
    )


class ManualMutationOut(BaseModel):
    """一条人工指定的单点突变。"""

    position: int
    wild_type: str
    mutant: str
    label: str
    sequence: str
    sequence_id: str
    sequence_length: int
    note: str = ""


class ManualBlockedOut(BaseModel):
    """被拒绝的目标位点。"""

    position: int
    residue: str
    reason: str


class ManualPlanOut(BaseModel):
    """人工突变清单预览结果。"""

    length: int
    name: str
    target_positions: list[int] = Field(default_factory=list)
    locked_positions: list[int] = Field(default_factory=list)
    mutation_count: int = 0
    position_count: int = 0
    mutations: list[ManualMutationOut] = Field(default_factory=list)
    blocked: list[ManualBlockedOut] = Field(default_factory=list)
    rejected_substitutions: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExportRequestIn(BaseModel):
    """候选导出请求。"""

    format: Literal["csv", "markdown"] = "csv"
    top_n: int = Field(default=20, ge=1, le=500)
    include_dimensions: bool = True
    title: str = "突变设计候选方案"
