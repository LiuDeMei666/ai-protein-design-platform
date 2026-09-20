"""折叠可行性评估。

三条独立证据
------------
1. **野生型 pLDDT 先验**：位点越刚性（pLDDT 越高），说明该处在天然结构中越被
   精确确定，替换造成构象代价的概率越大。反之柔性/低置信区更容易容忍突变。
2. **模型耐受度**：ESM-2 的 ΔlogP 归一化值（与稳定性维度同源，但此处作为
   "序列是否还能落在可折叠空间"的证据）。
3. **二级结构上下文**：螺旋/折叠中的位点对替换更敏感；无规卷曲区最宽容。

无结构信息时，第 1、3 条自动变为"不可用"并从权重中剔除（重新归一化），
总分仍可给出，但会在说明里标注置信度下降。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ....core.config import load_platform_config
from ....core.logging import get_logger
from ..scanner import PositionContext
from .zero_shot import ZeroShotScores

logger = get_logger(__name__)


@dataclass
class FoldabilityAssessment:
    """折叠可行性评估结果。"""

    score: float
    rationale: str
    evidence: dict[str, Any]


def tolerance_from_plddt(plddt: float | None) -> float | None:
    """由野生型 pLDDT 推断"可容忍突变程度"（0-100）。

    pLDDT 45 以下（无序/柔性）→ 约 95 分；90 以上（刚性核心）→ 约 36 分。
    """
    if plddt is None:
        return None
    return float(np.clip(95.0 - (plddt - 45.0) * 1.30, 20.0, 95.0))


def tolerance_from_secondary_structure(structure: str | None) -> float | None:
    """二级结构上下文的可容忍度。"""
    if structure is None:
        return None
    return {
        "H": 45.0,   # α-螺旋：主链氢键网络规则，替换易破坏
        "G": 50.0,   # 3-10 螺旋
        "I": 50.0,   # π-螺旋
        "E": 35.0,   # β-折叠：侧链参与疏水核心，替换代价通常最大
        "-": 82.0,   # 无规卷曲/环区
    }.get(structure, 60.0)


def assess_foldability(
    sequence: str,
    position: int,
    mutant: str,
    tolerance_score: float | None,
    context: PositionContext | None = None,
) -> FoldabilityAssessment:
    """评估某突变后序列的可折叠性。

    Args:
        tolerance_score: 模型耐受度分（0-100），来自 :mod:`zero_shot` 的
            ``stability_score``。**显式传数值而非 ZeroShotScores 对象**，
            是为了让"单点评估"与"组合精评"两条路径共用同一个函数——
            否则两条路径会因为折叠项的取舍不同而给出不一致的分数
            （同一突变在单点列表中 70.37 分、在组合列表中 76.48 分）。
    """
    config = load_platform_config().get("design", {}).get("foldability", {})
    w_plddt = float(config.get("plddt_weight", 0.45))
    w_tolerance = float(config.get("tolerance_weight", 0.35))
    w_structure = float(config.get("ss_weight", 0.20))

    if tolerance_score is None:
        tolerance_score = 50.0

    plddt_score = tolerance_from_plddt(context.plddt if context else None)
    structure_score = tolerance_from_secondary_structure(
        context.secondary_structure if context else None
    )

    parts: list[tuple[float, float]] = [(w_tolerance, float(tolerance_score))]
    if plddt_score is not None:
        parts.append((w_plddt, float(plddt_score)))
    if structure_score is not None:
        parts.append((w_structure, float(structure_score)))

    total_weight = sum(weight for weight, _ in parts)
    score = sum(weight * value for weight, value in parts) / max(1e-6, total_weight)

    available: list[str] = ["模型耐受度"]
    missing: list[str] = []
    if plddt_score is not None:
        available.append(f"野生型 pLDDT（{context.plddt:.0f}）")
    else:
        missing.append("野生型 pLDDT")
    if structure_score is not None:
        available.append(f"二级结构上下文（{context.secondary_structure}）")
    else:
        missing.append("二级结构上下文")

    rationale = (
        f"依据：{'、'.join(available)}，得分 {score:.1f}。"
        + (
            f"（缺少 {'、'.join(missing)}，相关证据已从权重中剔除）"
            if missing
            else ""
        )
    )

    return FoldabilityAssessment(
        score=round(float(score), 2),
        rationale=rationale,
        evidence={
            "tolerance_from_model": round(float(tolerance_score), 2),
            "tolerance_from_plddt": round(plddt_score, 2) if plddt_score is not None else None,
            "tolerance_from_secondary_structure": (
                round(structure_score, 2) if structure_score is not None else None
            ),
            "wild_type_plddt": context.plddt if context else None,
            "secondary_structure": context.secondary_structure if context else None,
            "missing_evidence": missing,
        },
    )


def delta_g_proxy(delta_logp: float | None, temperature_scale: float = 1.0) -> float | None:
    """把 ΔlogP 粗略换算为 ΔΔG 量级的代理值（kcal/mol）。

    **重要说明**：这不是经过校准的 ΔΔG 预测器。蛋白质语言模型的 ΔlogP 与实验
    ΔΔG 存在单调相关，但需要实测数据做线性/等温回归校准后才能给出 kcal/mol 量纲。
    此处仅提供一个便于实验人员建立直觉的粗尺度代理，报告中会明确标注
    "未校准"，企业数据到位后可用 :mod:`backend.app.ml` 做正式校准。
    """
    if delta_logp is None:
        return None
    # 经验粗尺度：ΔlogP 每变化 1 约对应 0.5-1 kcal/mol 量级
    return round(delta_logp * 0.7 * temperature_scale, 2)
