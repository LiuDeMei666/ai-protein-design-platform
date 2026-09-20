"""序列相关的请求 / 响应模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ProteinType = Literal["collagen", "protease", "protein_a", "generic"]

PROTEIN_TYPE_LABELS: dict[str, str] = {
    "collagen": "胶原蛋白",
    "protease": "重组蛋白酶",
    "protein_a": "蛋白 A",
    "generic": "通用蛋白",
}


class SequenceValidateIn(BaseModel):
    """校验序列请求。"""

    sequence: str = Field(description="氨基酸序列或 FASTA 文本")
    strict: bool = Field(
        default=False,
        description="为 True 时歧义残基视为错误（突变设计前置校验使用）",
    )

    @field_validator("sequence")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("序列不能为空")
        return value


class SequenceValidateOut(BaseModel):
    """校验结果。"""

    ok: bool
    design_ready: bool
    sequence: str
    length: int
    sha256: str
    invalid_chars: dict[str, int] = Field(default_factory=dict)
    ambiguous_positions: list[int] = Field(default_factory=list)
    ambiguous_summary: dict[str, int] = Field(default_factory=dict)
    composition: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    protein_type_labels: dict[str, str] = Field(default_factory=lambda: dict(PROTEIN_TYPE_LABELS))


class FastaParseIn(BaseModel):
    """FASTA 解析请求。"""

    text: str


class FastaRecordOut(BaseModel):
    """FASTA 记录。"""

    name: str
    header: str
    length: int


class SequenceCreateIn(BaseModel):
    """创建序列（落库）。"""

    project_id: int
    name: str = Field(min_length=1, max_length=200)
    sequence: str
    protein_type: ProteinType = "generic"
    note: str | None = None


class SequenceOut(BaseModel):
    """序列详情。"""

    id: int
    project_id: int
    name: str
    protein_type: str
    protein_type_label: str = ""
    sequence: str
    length: int
    sha256: str
    note: str | None = None
    created_at: datetime | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ProjectCreateIn(BaseModel):
    """创建项目。"""

    name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    host_system: str = "ecoli"


class ProjectOut(BaseModel):
    """项目详情。"""

    id: int
    name: str
    description: str | None = None
    host_system: str
    created_at: datetime | None = None
    sequence_count: int = 0
    design_count: int = 0
    experiment_count: int = 0
