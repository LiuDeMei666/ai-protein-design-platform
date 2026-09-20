"""基础生物物理量计算（确定性，不依赖任何模型）。

这些量是**完全可复现**的：同样的序列永远得到同样的数值，因此在报告里作为
"无争议的基线证据"，与模型预测的（带置信度的）指标区分开。

实现基于 Biopython ``ProtParam``（Guruprasad 不稳定指数、Bjellqvist pI 等经典
算法）+ 本项目的序列特征工具，互为交叉校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.logging import get_logger
from ..sequence.feature_utils import (
    ALIPHATIC_AA,
    AROMATIC_AA,
    HYDROPHOBIC_AA,
    POLAR_AA,
    POSITIVE_AA,
    NEGATIVE_AA,
    charge_profile,
    composition,
    composition_counts,
    net_charge,
)

logger = get_logger(__name__)

STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: N 端规则（Bachmair / Varshavsky）：Met 之后的第二个残基决定蛋白半衰期
N_END_RULE: dict[str, str] = {
    "A": "stabilizing", "C": "stabilizing", "G": "stabilizing", "P": "stabilizing",
    "S": "stabilizing", "T": "stabilizing", "V": "stabilizing", "M": "stabilizing",
    "R": "destabilizing", "K": "destabilizing", "F": "destabilizing", "L": "destabilizing",
    "W": "destabilizing", "Y": "destabilizing", "I": "destabilizing",
    "D": "destabilizing", "E": "destabilizing", "N": "destabilizing", "Q": "destabilizing",
    "H": "neutral",
}


def sanitize_sequence(sequence: str) -> tuple[str, int]:
    """剔除非标准氨基酸，返回 (可用序列, 被剔除数量)。

    Biopython ``ProteinAnalysis`` 只接受 20 种标准残基；歧义残基（B/Z/X/U/O）
    会被剔除并计数，而不是让整条链路报错。
    """
    cleaned = "".join(char for char in sequence if char in STANDARD_AA)
    return cleaned, len(sequence) - len(cleaned)


@dataclass
class BiophysResult:
    """基础生物物理量。"""

    values: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"values": self.values, "warnings": self.warnings}


def compute_biophys(sequence: str) -> BiophysResult:
    """计算全部基础生物物理量。"""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    warnings: list[str] = []
    cleaned, dropped = sanitize_sequence(sequence)
    if dropped:
        warnings.append(
            f"有 {dropped} 个非标准/歧义残基未参与生物物理量计算，相关指标置信度下降。"
        )
    if not cleaned:
        return BiophysResult(values={"length": len(sequence), "computable": False}, warnings=warnings)

    analysis = ProteinAnalysis(cleaned)
    counts = composition_counts(cleaned)
    ratios = composition(cleaned)
    length = len(cleaned)

    # --- 电荷曲线与等电点（本项目实现，与 Biopython 交叉校验）---
    profile = charge_profile(cleaned)
    computed_pi = profile.isoelectric_point

    try:
        biopython_pi = float(analysis.isoelectric_point())
    except Exception:  # pragma: no cover - 极短序列可能失败
        biopython_pi = computed_pi

    # --- 二级结构组成（Chou-Fasman 倾向，无结构时的兜底）---
    try:
        helix, turn, sheet = (float(value) for value in analysis.secondary_structure_fraction())
    except Exception:  # pragma: no cover
        helix = turn = sheet = 0.0

    try:
        extinction_reduced, extinction_cystine = (
            float(value) for value in analysis.molar_extinction_coefficient()
        )
    except Exception:  # pragma: no cover
        extinction_reduced = extinction_cystine = 0.0

    positive = sum(counts[aa] for aa in POSITIVE_AA)
    negative = sum(counts[aa] for aa in NEGATIVE_AA)
    hydrophobic = sum(counts[aa] for aa in HYDROPHOBIC_AA)
    polar = sum(counts[aa] for aa in POLAR_AA)
    aromatic = sum(counts[aa] for aa in AROMATIC_AA)
    aliphatic = sum(counts[aa] for aa in ALIPHATIC_AA)

    instability = float(analysis.instability_index())
    if instability > 40:
        warnings.append(
            f"不稳定指数 {instability:.1f} > 40，按 Guruprasad 判据该蛋白在体外可能不稳定。"
        )

    values: dict[str, Any] = {
        "computable": True,
        "length": length,
        "input_length": len(sequence),
        "dropped_residues": dropped,
        "molecular_weight_da": round(float(analysis.molecular_weight()), 1),
        "molecular_weight_kda": round(float(analysis.molecular_weight()) / 1000.0, 2),
        "isoelectric_point": round(biopython_pi, 2),
        "isoelectric_point_computed": computed_pi,
        "gravy": round(float(analysis.gravy()), 4),
        "aromaticity": round(float(analysis.aromaticity()), 4),
        "instability_index": round(instability, 2),
        "instability_class": "稳定" if instability <= 40 else "可能不稳定",
        "molar_extinction_coefficient_reduced": round(extinction_reduced),
        "molar_extinction_coefficient_cystine": round(extinction_cystine),
        "secondary_structure_fraction_chou_fasman": {
            "helix": round(helix, 4),
            "turn": round(turn, 4),
            "sheet": round(sheet, 4),
        },
        "net_charge_ph7": round(net_charge(cleaned, 7.0), 3),
        "net_charge_ph5": round(net_charge(cleaned, 5.0), 3),
        "net_charge_ph9": round(net_charge(cleaned, 9.0), 3),
        "charge_profile": {
            "ph": profile.ph_values[::4],
            "charge": [round(value, 3) for value in profile.charges[::4]],
        },
        "counts": counts,
        "ratios": {key: round(value, 4) for key, value in ratios.items()},
        "group_ratios": {
            "positive": round(positive / length, 4),
            "negative": round(negative / length, 4),
            "charged": round((positive + negative) / length, 4),
            "hydrophobic": round(hydrophobic / length, 4),
            "polar": round(polar / length, 4),
            "aromatic": round(aromatic / length, 4),
            "aliphatic": round(aliphatic / length, 4),
        },
        "special_residues": {
            "cys": counts["C"],
            "trp": counts["W"],
            "met": counts["M"],
            "pro": counts["P"],
            "gly": counts["G"],
            "his": counts["H"],
            "lys": counts["K"],
            "arg": counts["R"],
            "asn": counts["N"],
            "gln": counts["Q"],
            "asp": counts["D"],
            "glu": counts["E"],
        },
        "n_end_rule": {
            "second_residue": cleaned[1] if len(cleaned) > 1 else "",
            "class": N_END_RULE.get(cleaned[1], "unknown") if len(cleaned) > 1 else "unknown",
        },
        "algorithm": {
            "molecular_weight": "Biopython ProtParam（平均同位素质量）",
            "isoelectric_point": "Bjellqvist 平均 pKa 集 + 曲线过零插值",
            "gravy": "Kyte-Doolittle 疏水性标度均值",
            "instability_index": "Guruprasad 1990 二肽权重法",
            "secondary_structure_fraction": "Chou-Fasman 倾向",
            "charge_profile": "Henderson-Hasselbalch（本项目实现）",
        },
    }

    return BiophysResult(values=values, warnings=warnings)


def stability_hint_from_biophys(values: dict[str, Any]) -> dict[str, float]:
    """从生物物理量抽取稳定性相关特征（供 :mod:`stability` 使用）。"""
    ratios = values.get("group_ratios", {})
    fractions = values.get("secondary_structure_fraction_chou_fasman", {})
    return {
        "charged_fraction": float(ratios.get("charged", 0.0)),
        "hydrophobic_fraction": float(ratios.get("hydrophobic", 0.0)),
        "instability_index": float(values.get("instability_index", 40.0)),
        "helix_fraction": float(fractions.get("helix", 0.0)),
        "gravy": float(values.get("gravy", 0.0)),
        # IVYWREL 占比：已知与嗜热性正相关的经验特征（Ile/Val/Tyr/Trp/Arg/Glu/Leu）
        "ivywrel": _ivywrel(values.get("counts", {}), int(values.get("length", 1)) or 1),
    }


def _ivywrel(counts: dict[str, int], length: int) -> float:
    return round(sum(counts.get(aa, 0) for aa in "IVYWREL") / length, 4)
