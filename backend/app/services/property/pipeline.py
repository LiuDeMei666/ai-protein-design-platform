"""理化性质预测流水线。

编排顺序::

    序列校验
      -> 基础生物物理量（确定性，无模型）
      -> [可选] 结构统计（pLDDT / SASA / 疏水暴露）
      -> [可选] ESM-2 序列自然度
      -> 9 项指标评估
      -> 汇总（雷达图数据 + 风险计数）

设计取舍：ESM-2 与结构都是**可选增强项**。缺任一或全部时，流水线仍能给出完整的
9 项指标（相应证据项标注为"不可用"并在权重中自动剔除），保证平台在断网、
无 GPU 的前提下依然可用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.logging import describe_sequence, get_logger
from ..sequence.validator import SequenceCheck, validate_sequence
from .aggregation import assess_aggregation
from .biophys import compute_biophys
from .expression import assess_expression
from .immunogenicity import assess_immunogenicity
from .metric import Metric, build_summary
from .protease_sites import analyze_protease_sites, assess_protease_resistance
from .ptm_sites import analyze_ptm_sites, assess_ptm_risk
from .solubility import assess_solubility
from .stability import assess_acid_stability, assess_alkali_stability, assess_thermostability

logger = get_logger(__name__)

METRIC_ORDER: tuple[str, ...] = (
    "thermostability",
    "alkali_stability",
    "acid_stability",
    "solubility",
    "aggregation",
    "expression",
    "ptm_sites",
    "immunogenicity",
    "protease_resistance",
)


@dataclass
class PropertyReport:
    """完整性质报告。"""

    sequence: str
    length: int
    protein_type: str
    host_system: str
    biophys: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    esm_used: bool = False
    structure_used: bool = False

    def metric(self, key: str) -> Metric | None:
        return self.metrics.get(key)

    def to_dict(self, include_tracks: bool = True) -> dict[str, Any]:
        """序列化为 API 响应。

        Args:
            include_tracks: 是否包含逐残基轨道数据（聚集倾向轨道体积较大）。
        """
        payload_metrics: dict[str, Any] = {}
        for key in METRIC_ORDER:
            metric = self.metrics.get(key)
            if metric is None:
                continue
            item = metric.to_dict()
            if not include_tracks:
                item["meta"].pop("profile", None)
                item["locus"] = item["locus"][:50]
            payload_metrics[key] = item

        return {
            "length": self.length,
            "protein_type": self.protein_type,
            "host_system": self.host_system,
            "biophys": self.biophys,
            "metrics": payload_metrics,
            "summary": self.summary,
            "warnings": self.warnings,
            "esm_used": self.esm_used,
            "structure_used": self.structure_used,
        }


def _try_esm_naturalness(sequence: str) -> tuple[float | None, str | None]:
    """尝试计算 ESM-2 自然度；失败时返回 (None, 原因) 而不是抛错。"""
    try:
        from ..embedding.esm2 import get_service

        service = get_service()
        service.ensure_loaded()
        return service.sequence_naturalness(sequence), None
    except Exception as exc:
        logger.warning("ESM-2 自然度计算失败，该项将从总分中剔除: %s", exc)
        return None, str(exc)


def predict_properties(
    sequence: str,
    *,
    protein_type: str = "generic",
    host_system: str = "ecoli",
    structure_stats: dict[str, Any] | None = None,
    dna_sequence: str | None = None,
    use_esm: bool = True,
    validator_result: SequenceCheck | None = None,
) -> PropertyReport:
    """执行完整性质预测。

    Args:
        sequence: 氨基酸序列。
        protein_type: 蛋白类型（collagen/protease/protein_a/generic），影响胶原羟化位点识别。
        host_system: 宿主表达体系（决定 CAI 参考表）。
        structure_stats: 结构统计字典（``StructureResult.stats``），可选。
        dna_sequence: 基因序列，提供时额外计算真实 CAI。
        use_esm: 是否使用 ESM-2 计算序列自然度（需要 GPU 与已下载权重）。
        validator_result: 已完成的序列校验结果，避免重复校验。
    """
    from ...core.errors import SequenceError

    check = validator_result or validate_sequence(sequence)
    if not check.ok:
        raise SequenceError("; ".join(check.errors), detail={"warnings": check.warnings})

    cleaned = check.sequence
    logger.info(
        "性质预测开始 %s type=%s host=%s structure=%s esm=%s",
        describe_sequence(cleaned),
        protein_type,
        host_system,
        bool(structure_stats),
        use_esm,
    )

    warnings = list(check.warnings)

    # 0) 占位结构不得参与评分
    #    stub Provider 生成的骨架 pLDDT 被刻意压低（30-45），若计入会把真实序列的
    #    热稳定性/酸稳定性/聚集风险无端拉低，属于"用一个假结构去惩罚真序列"。
    if structure_stats and structure_stats.get("stub"):
        warnings.append(
            "检测到占位结构（stub）的统计量，已将其从评分中剔除，"
            "以免用无生物学意义的骨架拉低真实序列的评分。"
        )
        structure_stats = None

    # 1) 基础生物物理量
    biophys_result = compute_biophys(cleaned)
    values = biophys_result.values
    warnings.extend(biophys_result.warnings)

    # 2) ESM-2 自然度（可选）
    naturalness: float | None = None
    if use_esm:
        naturalness, esm_error = _try_esm_naturalness(cleaned)
        if esm_error:
            warnings.append(
                f"ESM-2 序列自然度不可用（{esm_error}），热稳定性评分已自动剔除该项。"
            )

    # 3) 各项指标
    metrics: dict[str, Metric] = {}
    metrics["thermostability"] = assess_thermostability(
        cleaned, values, structure_stats=structure_stats, esm_naturalness=naturalness
    )
    metrics["alkali_stability"] = assess_alkali_stability(cleaned, values)
    metrics["acid_stability"] = assess_acid_stability(cleaned, values, structure_stats=structure_stats)
    metrics["solubility"] = assess_solubility(cleaned, values)
    metrics["aggregation"] = assess_aggregation(cleaned, values, structure_stats=structure_stats)
    metrics["expression"] = assess_expression(cleaned, values, host_system=host_system, dna_sequence=dna_sequence)
    metrics["ptm_sites"] = assess_ptm_risk(cleaned, analyze_ptm_sites(cleaned, protein_type), protein_type)
    metrics["immunogenicity"] = assess_immunogenicity(cleaned, values, structure_stats=structure_stats)
    metrics["protease_resistance"] = assess_protease_resistance(
        cleaned, analyze_protease_sites(cleaned)
    )

    summary = build_summary(metrics)
    summary["protein_type"] = protein_type
    summary["host_system"] = host_system
    summary["esm_used"] = naturalness is not None
    summary["structure_used"] = bool(structure_stats)

    report = PropertyReport(
        sequence=cleaned,
        length=len(cleaned),
        protein_type=protein_type,
        host_system=host_system,
        biophys=values,
        metrics=metrics,
        summary=summary,
        warnings=warnings,
        esm_used=naturalness is not None,
        structure_used=bool(structure_stats),
    )

    logger.info(
        "性质预测完成 %s 综合分=%.1f 最弱项=%s",
        describe_sequence(cleaned),
        summary.get("overall_score", 0.0),
        [item["key"] for item in summary.get("weakest_metrics", [])],
    )
    return report


def predict_properties_summary_only(**kwargs: Any) -> dict[str, Any]:
    """仅返回汇总（不包含逐残基轨道），用于列表页。"""
    report = predict_properties(**kwargs)
    return report.to_dict(include_tracks=False)
