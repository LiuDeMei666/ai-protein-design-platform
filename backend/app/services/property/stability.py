"""蛋白耐受性评估：热稳定性、碱稳定性、酸稳定性。

算法定位
--------
需求文档要求预测"蛋白耐受性（碱稳定性、热稳定性、酸稳定性等）"。在**没有企业
标注数据**的前提下，本项目采用"已发表的序列/结构经验特征 + 规则明确的水解敏感
基序识别"组合，而不是拟合一个无数据支撑的黑箱模型。每一项都给出算法来源与
依据，便于实验人员判断可信度。

热稳定性：采用嗜热蛋白的公认序列特征（IVYWREL 占比、带电残基占比、脯氨酸/甘氨酸
含量）+ Guruprasad 不稳定指数 + 可选的结构置信度与蛋白语言模型自然度。

碱稳定性：碱性条件下的主要失活途径是 **Asn/Gln 脱酰胺**（尤其 Asn-Gly）与
**二硫键错配/断裂**，因此以这两类基序密度为核心。

酸稳定性：酸性条件下的主要风险是 **Asp-Pro 肽键酸水解** 与酸性残基质子化导致的
盐桥丧失，同时芳香族残基含量与已知酸稳定蛋白（如胃蛋白酶）正相关。
"""

from __future__ import annotations

from typing import Any

from ...core.logging import get_logger
from .metric import Evidence, Metric, aggregate_weighted, linear_score, peak_score

logger = get_logger(__name__)

#: 脱酰胺敏感基序（第二位残基的促进效应由强到弱）
DEAMIDATION_MOTIFS: dict[str, float] = {
    "NG": 1.00, "NS": 0.72, "NT": 0.62, "NH": 0.65, "ND": 0.55, "NA": 0.45,
    "QG": 0.32, "QS": 0.20, "QT": 0.18,
}
#: 酸敏感的 Asp-Pro 肽键
ACID_LABILE_MOTIFS: tuple[str, ...] = ("DP",)


def count_motifs(sequence: str, motifs: dict[str, float] | tuple[str, ...]) -> dict[str, int]:
    """统计基序出现次数（不重叠）。"""
    keys = motifs.keys() if isinstance(motifs, dict) else motifs
    counts: dict[str, int] = {}
    for motif in keys:
        total = 0
        start = 0
        while True:
            index = sequence.find(motif, start)
            if index < 0:
                break
            total += 1
            start = index + len(motif)
        counts[motif] = total
    return counts


def weighted_motif_density(
    sequence: str, motifs: dict[str, float], per: int = 100
) -> tuple[float, list[dict[str, Any]]]:
    """加权基序密度（每 ``per`` 个残基），并返回位点明细。"""
    counts = count_motifs(sequence, motifs)
    loci: list[dict[str, Any]] = []
    weighted = 0.0
    for motif, count in counts.items():
        weight = motifs[motif]
        weighted += count * weight
        for index in range(len(sequence)):
            if sequence.startswith(motif, index):
                loci.append(
                    {
                        "position": index,
                        "residue": sequence[index],
                        "motif": motif,
                        "severity": round(weight, 2),
                        "type": "deamidation",
                    }
                )
    density = weighted / max(1, len(sequence)) * per
    return round(density, 3), loci


# --------------------------------------------------------------------------- #
# 热稳定性
# --------------------------------------------------------------------------- #
def assess_thermostability(
    sequence: str,
    biophys_values: dict[str, Any],
    structure_stats: dict[str, Any] | None = None,
    esm_naturalness: float | None = None,
) -> Metric:
    """热稳定性评估。

    Args:
        esm_naturalness: 序列在 ESM-2 下的平均野生型对数概率（越接近 0 越"自然"）。
            为 ``None`` 时该项不计入总分。
    """
    length = max(1, len(sequence))
    counts = biophys_values.get("counts", {})
    groups = biophys_values.get("group_ratios", {})
    instability = float(biophys_values.get("instability_index", 40.0))

    ivywrel = sum(counts.get(aa, 0) for aa in "IVYWREL") / length
    charged = float(groups.get("charged", 0.0))
    pro_fraction = counts.get("P", 0) / length
    gly_fraction = counts.get("G", 0) / length

    mean_plddt: float | None = None
    if structure_stats:
        raw = structure_stats.get("mean_plddt")
        if raw is None:
            bands = structure_stats.get("plddt_bands", {})
            # 没有均值时用分级占比估算
            if bands:
                mean_plddt = (
                    bands.get("very_high", 0) * 95
                    + bands.get("confident", 0) * 80
                    + bands.get("low", 0) * 60
                    + bands.get("very_low", 0) * 35
                )
        else:
            mean_plddt = float(raw)

    parts = [
        (
            0.25,
            linear_score(ivywrel, worst=0.28, best=0.45),
            Evidence(
                label="IVYWREL 占比（嗜热性相关特征）",
                value=round(ivywrel, 4),
                rationale=(
                    "Ile/Val/Tyr/Trp/Arg/Glu/Leu 占比是公认的嗜热蛋白序列特征，"
                    "嗜热蛋白通常 >0.40，常温蛋白约 0.30-0.35。"
                ),
            ),
        ),
        (
            0.15,
            linear_score(charged, worst=0.16, best=0.32),
            Evidence(
                label="带电残基占比（DEKR）",
                value=round(charged, 4),
                rationale="较高的表面带电残基比例有利于形成盐桥网络，提升热稳定性。",
            ),
        ),
        (
            0.15,
            linear_score(instability, worst=55.0, best=20.0),
            Evidence(
                label="不稳定指数（Guruprasad）",
                value=round(instability, 2),
                rationale="<40 判为稳定；该指数基于二肽权重，反映体外稳定性倾向。",
            ),
        ),
        (
            0.08,
            peak_score(gly_fraction, optimum=0.055, tolerance=0.075),
            Evidence(
                label="甘氨酸含量",
                value=round(gly_fraction, 4),
                rationale="甘氨酸提高主链柔性、增加构象熵，过量（>10%）通常降低热稳定性。",
            ),
        ),
        (
            0.07,
            peak_score(pro_fraction, optimum=0.055, tolerance=0.07),
            Evidence(
                label="脯氨酸含量",
                value=round(pro_fraction, 4),
                rationale="适度脯氨酸可刚性化环区、降低去折叠熵；过高则破坏二级结构。",
            ),
        ),
        (
            0.15,
            linear_score(mean_plddt, worst=60.0, best=95.0) if mean_plddt is not None else None,
            Evidence(
                label="结构平均置信度（pLDDT）",
                value=round(mean_plddt, 2) if mean_plddt is not None else "不可用",
                rationale="高置信度的刚性结构通常对应更好的热稳定性；无结构时为不可用项。",
            ),
        ),
        (
            0.15,
            linear_score(esm_naturalness, worst=-5.0, best=-1.2)
            if esm_naturalness is not None
            else None,
            Evidence(
                label="ESM-2 序列自然度",
                value=round(esm_naturalness, 3) if esm_naturalness is not None else "不可用",
                rationale=(
                    "野生型残基在 ESM-2 下的平均对数概率。数值越接近 0 说明该序列越符合"
                    "自然蛋白的统计规律；显著偏离常见于人工设计或不稳定序列。"
                ),
            ),
        ),
    ]

    score, evidence = aggregate_weighted(parts)
    return Metric(
        key="thermostability",
        label="热稳定性",
        score=round(score, 1),
        algorithm=(
            "IVYWREL 特征 + 带电残基占比 + Guruprasad 不稳定指数 + 甘氨酸/脯氨酸最优化"
            " + 可选 pLDDT 与 ESM-2 自然度"
        ),
        rationale=(
            "基于已发表的嗜热蛋白序列特征与经典稳定性指标组合评分，"
            "不依赖企业实验数据即可给出可比性排序；精度以 ΔΔG 实测数据校准前为趋势级。"
        ),
        confidence=0.55 if esm_naturalness is None else 0.65,
        evidence=evidence,
        meta={"ivywrel": round(ivywrel, 4), "charged_fraction": round(charged, 4)},
    )


# --------------------------------------------------------------------------- #
# 碱稳定性
# --------------------------------------------------------------------------- #
def assess_alkali_stability(sequence: str, biophys_values: dict[str, Any]) -> Metric:
    """碱稳定性评估：以脱酰胺敏感基序与二硫键为核心风险。"""
    length = max(1, len(sequence))
    counts = biophys_values.get("counts", {})

    density, deamidation_loci = weighted_motif_density(sequence, DEAMIDATION_MOTIFS)
    asn_fraction = counts.get("N", 0) / length
    gln_fraction = counts.get("Q", 0) / length
    cys_count = counts.get("C", 0)
    disulfides = cys_count // 2

    cys_loci = [
        {"position": index, "residue": "C", "type": "cysteine", "severity": 1.0}
        for index, char in enumerate(sequence)
        if char == "C"
    ]

    parts = [
        (
            0.40,
            linear_score(density, worst=3.0, best=0.0),
            Evidence(
                label="脱酰胺敏感基序密度（每 100 残基）",
                value=density,
                rationale=(
                    "Asn-Gly 是脱酰胺速率最高的基序（相对权重 1.00），其后依次是 "
                    "Asn-Ser/Asn-His/Asn-Thr；碱性 pH 下 Asn/Gln 侧链酰胺基被 OH⁻ 攻击"
                    "生成 Asp/Glu，是碱性条件下失活的主因。"
                ),
            ),
        ),
        (
            0.18,
            linear_score(asn_fraction, worst=0.09, best=0.015),
            Evidence(
                label="天冬酰胺占比",
                value=round(asn_fraction, 4),
                rationale="Asn 总量越高，可供脱酰胺的位点越多。",
            ),
        ),
        (
            0.12,
            linear_score(gln_fraction, worst=0.07, best=0.012),
            Evidence(
                label="谷氨酰胺占比",
                value=round(gln_fraction, 4),
                rationale="Gln 脱酰胺速率远低于 Asn，但在长时间碱性处理下仍会累积。",
            ),
        ),
        (
            0.30,
            linear_score(float(disulfides), worst=6.0, best=0.0),
            Evidence(
                label="潜在二硫键数量（Cys 对数）",
                value=disulfides,
                rationale=(
                    "碱性条件下二硫键易发生 β-消除与重排（scrambling），"
                    "含多个二硫键的蛋白在 pH>9 时构象稳定性显著下降。"
                ),
            ),
        ),
    ]

    score, evidence = aggregate_weighted(parts)
    loci = deamidation_loci + cys_loci

    return Metric(
        key="alkali_stability",
        label="碱稳定性",
        score=round(score, 1),
        algorithm="脱酰胺敏感基序加权密度 + Asn/Gln 占比 + 二硫键负荷",
        rationale=(
            "碱性失活以 Asn/Gln 脱酰胺与二硫键重排为主。"
            "改造建议集中在消除 Asn-Gly/Asn-Ser 基序（如 N→Q/D/T）与减少非必需 Cys。"
        ),
        confidence=0.6,
        evidence=evidence,
        locus=loci[:400],
        meta={"deamidation_density": density, "disulfides": disulfides},
    )


# --------------------------------------------------------------------------- #
# 酸稳定性
# --------------------------------------------------------------------------- #
def assess_acid_stability(
    sequence: str,
    biophys_values: dict[str, Any],
    structure_stats: dict[str, Any] | None = None,
) -> Metric:
    """酸稳定性评估：Asp-Pro 酸水解、酸性残基质子化与芳香族含量。"""
    length = max(1, len(sequence))
    counts = biophys_values.get("counts", {})
    groups = biophys_values.get("group_ratios", {})

    dp_counts = count_motifs(sequence, ACID_LABILE_MOTIFS)
    dp_count = dp_counts.get("DP", 0)
    dp_density = dp_count / length * 100

    acidic = float(groups.get("negative", 0.0))
    aromatic = float(groups.get("aromatic", 0.0))
    deamidation_density, deamidation_loci = weighted_motif_density(sequence, DEAMIDATION_MOTIFS)

    dp_loci = [
        {"position": index, "residue": "D", "motif": "DP", "type": "acid_labile_bond", "severity": 1.0}
        for index in range(len(sequence) - 1)
        if sequence.startswith("DP", index)
    ]

    buried_hydrophobic: float | None = None
    if structure_stats:
        exposure = structure_stats.get("hydrophobic_exposure", {})
        if isinstance(exposure, dict) and exposure.get("exposed_ratio") is not None:
            # 疏水核心暴露越少，抗酸去折叠能力越强
            buried_hydrophobic = 1.0 - float(exposure["exposed_ratio"])

    parts = [
        (
            0.32,
            linear_score(dp_density, worst=1.6, best=0.0),
            Evidence(
                label="Asp-Pro 肽键密度（每 100 残基）",
                value=round(dp_density, 3),
                rationale=(
                    "Asp-Pro 是唯一在酸性条件下显著酸水解的肽键（pH 2-3 时数小时内可断裂），"
                    "是重组蛋白在低 pH 洗脱/酸沉步骤中的主要断链风险点。"
                ),
            ),
        ),
        (
            0.20,
            linear_score(acidic, worst=0.17, best=0.06),
            Evidence(
                label="酸性残基占比（DE）",
                value=round(acidic, 4),
                rationale="酸性残基在低 pH 下质子化、失去负电荷，导致盐桥网络瓦解与局部去折叠。",
            ),
        ),
        (
            0.20,
            linear_score(aromatic, worst=0.05, best=0.14),
            Evidence(
                label="芳香族残基占比（FWY）",
                value=round(aromatic, 4),
                rationale="芳香族残基的堆积作用在低 pH 下仍较稳定，已知酸稳定蛋白（如胃蛋白酶）芳香族含量偏高。",
            ),
        ),
        (
            0.16,
            linear_score(deamidation_density, worst=3.0, best=0.0),
            Evidence(
                label="脱酰胺敏感基序密度",
                value=deamidation_density,
                rationale="低 pH 同样会促进 Asn-Gly 脱酰胺（速率峰值约在 pH 5 与 pH 10）。",
            ),
        ),
        (
            0.12,
            linear_score(buried_hydrophobic, worst=0.6, best=0.95)
            if buried_hydrophobic is not None
            else None,
            Evidence(
                label="疏水核心埋藏程度",
                value=round(buried_hydrophobic, 4) if buried_hydrophobic is not None else "不可用",
                rationale="疏水核心埋藏良好者在酸诱导去折叠下更耐受；无结构时为不可用项。",
            ),
        ),
    ]

    score, evidence = aggregate_weighted(parts)
    return Metric(
        key="acid_stability",
        label="酸稳定性",
        score=round(score, 1),
        algorithm="Asp-Pro 酸敏感键密度 + 酸性残基占比 + 芳香族占比 + 疏水核心埋藏",
        rationale=(
            "酸性条件下以 Asp-Pro 肽键水解与电荷中和导致的盐桥瓦解为主。"
            "改造建议：消除非必需 Asp-Pro 键、在表面引入可维持低 pH 构象的芳香族/脯氨酸。"
        ),
        confidence=0.6,
        evidence=evidence,
        locus=(dp_loci + deamidation_loci)[:400],
        meta={"dp_motif_count": dp_count, "dp_density": round(dp_density, 3)},
    )
