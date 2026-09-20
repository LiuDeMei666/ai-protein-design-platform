"""结构预测统一契约。

为什么要做 Provider 抽象
------------------------
用户选定"在线 ESM Atlas"，但企业内网随时可能断网，且超长序列（胶原蛋白）会
触及在线接口的长度上限。因此把"结构来源"抽象成可替换的实现：

============  ==========================================================
Provider      说明
============  ==========================================================
esmatlas      在线 ESM Atlas REST（主通道，实测 2.26s/72aa，CC-BY-4.0 可商用）
local_esmfold 本地 ESMFold（V100 可跑，需额外权重，作为离线后备）
stub          离线占位：返回规则化假结构，保证界面与单测可用
============  ==========================================================

注册表按 ``auto`` 顺序探测可用性，任一实现失败**自动降级并明确标注原因**，
绝不静默返回空结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ...core.errors import PlatformError


@dataclass
class StructureResult:
    """结构预测结果。

    所有字段都会透传到前端与 API 响应中，其中 ``truncated`` / ``segments`` /
    ``degradation_reason`` 用于**显式标注**本次预测的局限，避免用户误判。
    """

    #: PDB 文本（可直接交给 3Dmol.js 渲染）
    pdb_text: str
    #: 逐残基 pLDDT（0-100），长度等于序列长度
    plddt: list[float]
    #: 平均 pLDDT
    mean_plddt: float
    #: 实际产出该结构的 Provider 名称
    source: str
    #: 是否发生了截断（有序列片段未被预测）
    truncated: bool = False
    #: 实际预测的片段区间（0-based 半开），完整预测时为 [(0, L)]
    segments: list[tuple[int, int]] = field(default_factory=list)
    #: 是否命中磁盘缓存
    from_cache: bool = False
    #: 降级原因（例如"主通道超时，已切换 stub"）
    degradation_reason: str | None = None
    #: Provider 侧模型版本标识
    model_version: str | None = None
    #: 结构统计：二级结构组成、回旋半径、疏水暴露、接触图等
    stats: dict[str, Any] = field(default_factory=dict)
    #: 面向用户的提示（非致命）
    warnings: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.plddt)

    def plddt_bands(self, very_high: float = 90, confident: float = 70, low: float = 50) -> dict[str, float]:
        """按 AlphaFold 官方分级统计 pLDDT 分布（占比）。"""
        total = len(self.plddt) or 1
        bands = {"very_high": 0, "confident": 0, "low": 0, "very_low": 0}
        for value in self.plddt:
            if value >= very_high:
                bands["very_high"] += 1
            elif value >= confident:
                bands["confident"] += 1
            elif value >= low:
                bands["low"] += 1
            else:
                bands["very_low"] += 1
        return {key: round(count / total, 4) for key, count in bands.items()}

    def to_summary(self) -> dict[str, Any]:
        """轻量摘要（不含 PDB 文本），用于列表与作业结果。"""
        return {
            "source": self.source,
            "model_version": self.model_version,
            "mean_plddt": round(self.mean_plddt, 2),
            "length": self.length,
            "truncated": self.truncated,
            "segments": [list(segment) for segment in self.segments],
            "from_cache": self.from_cache,
            "degradation_reason": self.degradation_reason,
            "warnings": self.warnings,
            "stats": self.stats,
        }


@runtime_checkable
class StructurePredictor(Protocol):
    """结构预测器契约。

    实现必须声明 :attr:`max_length`（``None`` 表示无硬限制），注册表据此决定
    是否需要先做域切分。
    """

    name: str
    max_length: int | None

    def available(self) -> bool:
        """当前环境下是否可用（不产生昂贵副作用）。"""
        ...

    def describe(self) -> dict[str, Any]:
        """可读状态描述，供健康检查与前端展示。"""
        ...

    def predict(self, sequence: str, *, allow_split: bool = True) -> StructureResult:
        """预测结构。实现须保证失败时抛出 :class:`PlatformError` 子类。"""
        ...


class BasePredictor:
    """Provider 公共基类：统一日志与未实现兜底。"""

    name: str = "base"
    max_length: int | None = None
    model_version: str | None = None

    def available(self) -> bool:  # pragma: no cover - 由子类覆盖
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_length": self.max_length,
            "available": False,
            "model_version": self.model_version,
            "reason": "未实现 describe()",
        }

    def predict(self, sequence: str, *, allow_split: bool = True) -> StructureResult:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _require_sequence(sequence: str) -> str:
        if not sequence:
            raise PlatformError("序列为空，无法进行结构预测")
        return sequence
