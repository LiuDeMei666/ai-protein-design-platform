"""结构预测相关的请求 / 响应模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class StructurePredictIn(BaseModel):
    """结构预测请求。"""

    sequence: str = Field(description="氨基酸序列（可含 FASTA 头）")
    protein_type: Literal["collagen", "protease", "protein_a", "generic"] = "generic"
    provider: str | None = Field(
        default=None,
        description="指定 Provider：esmatlas / local_esmfold / stub；留空则按降级链自动选择",
    )
    allow_split: bool = Field(default=True, description="超长序列是否允许自动分片折叠")
    use_cache: bool = Field(default=True, description="是否使用磁盘缓存")
    project_id: int | None = Field(default=None, description="归属项目；留空则使用默认项目")
    name: str | None = Field(default=None, description="序列名称（落库时使用）")


class ResiduePlddtOut(BaseModel):
    """逐残基置信度。"""

    index: int
    residue: str
    plddt: float


class StructureStatsOut(BaseModel):
    """结构统计信息。"""

    plddt_bands: dict[str, float] = Field(default_factory=dict)
    secondary_structure_composition: dict[str, float] = Field(default_factory=dict)
    radius_of_gyration: float = 0.0
    mean_relative_sasa: float = 0.0
    hydrophobic_exposure: dict[str, Any] = Field(default_factory=dict)
    flexible_region_ratio: float = 0.0
    algorithms: dict[str, str] = Field(default_factory=dict)
    elapsed_seconds: float | None = None


class StructureOut(BaseModel):
    """结构预测结果。

    ``pdb_text`` 与 ``plddt`` 体积较大，可通过 ``include_pdb=False`` 只取摘要。
    """

    source: str
    model_version: str | None = None
    mean_plddt: float
    length: int
    truncated: bool = False
    segments: list[list[int]] = Field(default_factory=list)
    from_cache: bool = False
    degradation_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    plddt: list[float] = Field(default_factory=list)
    secondary_structure: str = ""
    stats: StructureStatsOut | None = None
    pdb_text: str | None = Field(default=None, description="PDB 文本，include_pdb=False 时为 null")

    @classmethod
    def from_result(cls, result: Any, include_pdb: bool = True) -> "StructureOut":
        """由 :class:`StructureResult` 构建响应。"""
        stats = result.stats or {}
        return cls(
            source=result.source,
            model_version=result.model_version,
            mean_plddt=round(float(result.mean_plddt), 2),
            length=result.length,
            truncated=result.truncated,
            segments=[list(segment) for segment in result.segments],
            from_cache=result.from_cache,
            degradation_reason=result.degradation_reason,
            warnings=list(result.warnings),
            plddt=list(result.plddt),
            secondary_structure=str(stats.get("secondary_structure", "")),
            stats=StructureStatsOut(
                plddt_bands=stats.get("plddt_bands", {}),
                secondary_structure_composition=stats.get("secondary_structure_composition", {}),
                radius_of_gyration=stats.get("radius_of_gyration", 0.0),
                mean_relative_sasa=stats.get("mean_relative_sasa", 0.0),
                hydrophobic_exposure=stats.get("hydrophobic_exposure", {}),
                flexible_region_ratio=stats.get("flexible_region_ratio", 0.0),
                algorithms=stats.get("algorithms", {}),
                elapsed_seconds=stats.get("elapsed_seconds"),
            ),
            pdb_text=result.pdb_text if include_pdb else None,
        )


class ProviderStatusOut(BaseModel):
    """Provider 状态。"""

    name: str
    available: bool
    max_length: int | None = None
    model_version: str | None = None
    note: str | None = None
    reason: str | None = None
    install: str | None = None
