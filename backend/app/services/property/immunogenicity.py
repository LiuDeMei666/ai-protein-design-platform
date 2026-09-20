"""免疫原性风险评估（T 细胞表位筛选启发式）。

定位与边界（务必阅读）
----------------------
本模块是**筛查级启发式**，不是 NetMHCIIpan / IEDB 的替代品。它基于 MHC 分子
结合槽的**锚定残基偏好**这一公认结构原理：

* **MHC-II**：9 聚体核心，P1 为最主要锚点（偏好大疏水/芳香残基），
  P4/P6/P9 为次要锚点。
* **MHC-I**：8-10 聚体（此处固定取 9），P2 与 P9 为锚点。

打分方式是把各锚点的残基偏好相加得到相对的"结合倾向"，用于**同一平台内不同
候选之间的排序**，不能解释为绝对结合亲和力（IC50）。接口已预留，
企业若采购 NetMHCIIpan 授权，可替换 :func:`predict_mhc2_binders` 而不影响上层。

补充特征
--------
* **两亲性螺旋倾向**（Eisenberg 疏水矩）：能形成两亲性螺旋的片段更易被
  HLA-II 呈递，是实验验证过的免疫原性正相关特征。
* **疏水表面暴露**：结构可用时，表面暴露的疏水斑块与 B 细胞表位相关。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...core.config import load_platform_config
from ...core.logging import get_logger
from ..sequence.feature_utils import hydrophobic_moment
from .metric import Evidence, Metric, aggregate_weighted, linear_score

logger = get_logger(__name__)

#: MHC-II 锚点残基偏好（0-1，按已发表结合基序整理）
MHC2_ANCHORS: dict[int, dict[str, float]] = {
    1: {  # P1：最主要的锚点，偏好大疏水/芳香侧链
        "F": 1.00, "Y": 1.00, "W": 0.95, "L": 0.90, "I": 0.85,
        "V": 0.72, "M": 0.70, "A": 0.45, "T": 0.40, "P": 0.10,
        "D": 0.10, "E": 0.10, "K": 0.15, "R": 0.15,
    },
    4: {  # P4：偏好酸性或疏水
        "D": 0.90, "E": 0.90, "L": 0.72, "I": 0.72, "F": 0.70,
        "V": 0.65, "M": 0.60, "Y": 0.60, "A": 0.55, "K": 0.35,
        "R": 0.35, "P": 0.15,
    },
    6: {  # P6：偏好小残基
        "A": 0.90, "G": 0.85, "S": 0.85, "T": 0.80, "P": 0.80,
        "V": 0.75, "C": 0.65, "N": 0.60, "D": 0.55, "L": 0.45,
        "W": 0.25, "F": 0.30,
    },
    9: {  # P9：偏好碱性或疏水
        "K": 0.90, "R": 0.85, "L": 0.80, "I": 0.75, "F": 0.70,
        "V": 0.70, "A": 0.70, "M": 0.62, "Y": 0.55, "D": 0.35,
        "E": 0.35, "P": 0.20,
    },
}

#: MHC-I 锚点（9 聚体）
MHC1_P2: dict[str, float] = {
    "L": 1.00, "I": 0.92, "M": 0.90, "V": 0.85, "A": 0.72,
    "T": 0.70, "Q": 0.55, "S": 0.45, "F": 0.40, "P": 0.15, "D": 0.10, "E": 0.10,
}
MHC1_P9: dict[str, float] = {
    "V": 1.00, "L": 0.95, "I": 0.88, "A": 0.80, "M": 0.72,
    "T": 0.62, "K": 0.50, "R": 0.48, "F": 0.45, "D": 0.15, "E": 0.15,
}

ANCHOR_WEIGHTS = {1: 0.35, 4: 0.20, 6: 0.20, 9: 0.25}
DEFAULT_PREFERENCE = 0.25
#: 判定为"强结合表位"的阈值
STRONG_BINDER_THRESHOLD = 0.72


@dataclass
class Epitope:
    """一个预测的 T 细胞表位。"""

    start: int
    end: int
    peptide: str
    score: float
    mhc_class: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.start,
            "end": self.end,
            "residue": self.peptide[0] if self.peptide else "",
            "peptide": self.peptide,
            "peak": round(self.score, 3),
            "type": f"{self.mhc_class}_epitope",
            "severity": round(min(1.0, self.score), 2),
        }


def score_mhc2_core(core: str) -> float:
    """给一个 9 聚体核心打 MHC-II 结合倾向分（0-1）。"""
    if len(core) < 9:
        return 0.0
    total = 0.0
    for position, weight in ANCHOR_WEIGHTS.items():
        residue = core[position - 1]
        preference = MHC2_ANCHORS[position].get(residue, DEFAULT_PREFERENCE)
        total += weight * preference
    return round(total, 4)


def score_mhc1_core(core: str) -> float:
    """给一个 9 聚体打 MHC-I 结合倾向分（0-1）。"""
    if len(core) < 9:
        return 0.0
    p2 = MHC1_P2.get(core[1], 0.15)
    p9 = MHC1_P9.get(core[8], 0.15)
    return round(0.5 * p2 + 0.5 * p9, 4)


def predict_mhc2_binders(
    sequence: str, top_n: int = 10, threshold: float = STRONG_BINDER_THRESHOLD
) -> list[Epitope]:
    """滑动 9 聚体窗口，返回预测的 MHC-II 强结合表位。

    若要接入真实预测器（NetMHCIIpan / IEDB API），替换本函数即可，
    上层 :func:`assess_immunogenicity` 无需改动。
    """
    epitopes: list[Epitope] = []
    for start in range(0, max(0, len(sequence) - 8)):
        core = sequence[start : start + 9]
        score = score_mhc2_core(core)
        if score >= threshold:
            epitopes.append(
                Epitope(start=start, end=start + 9, peptide=core, score=score, mhc_class="mhc_ii")
            )
    epitopes.sort(key=lambda item: -item.score)
    return epitopes[:top_n]


def predict_mhc1_binders(sequence: str, top_n: int = 10) -> list[Epitope]:
    """滑动 9 聚体窗口，返回预测的 MHC-I 表位。"""
    epitopes: list[Epitope] = []
    for start in range(0, max(0, len(sequence) - 8)):
        core = sequence[start : start + 9]
        score = score_mhc1_core(core)
        epitopes.append(
            Epitope(start=start, end=start + 9, peptide=core, score=score, mhc_class="mhc_i")
        )
    epitopes.sort(key=lambda item: -item.score)
    return epitopes[:top_n]


def assess_immunogenicity(
    sequence: str,
    biophys_values: dict[str, Any] | None = None,
    structure_stats: dict[str, Any] | None = None,
) -> Metric:
    """免疫原性风险评估：分数越高表示风险越低。"""
    config = load_platform_config().get("immunogenicity", {})
    top_n = int(config.get("top_epitopes", 10))
    amphipathic_threshold = float(config.get("amphipathic_threshold", 0.45))

    length = max(1, len(sequence))
    mhc2_epitopes = predict_mhc2_binders(sequence, top_n=top_n)
    mhc1_binders = predict_mhc1_binders(sequence, top_n=top_n)

    # 全位点扫描用于统计"强结合窗口占比"
    strong_window_count = 0
    window_count = 0
    best_score = 0.0
    for start in range(0, max(0, len(sequence) - 8)):
        core = sequence[start : start + 9]
        score = score_mhc2_core(core)
        window_count += 1
        best_score = max(best_score, score)
        if score >= STRONG_BINDER_THRESHOLD:
            strong_window_count += 1
    strong_ratio = strong_window_count / window_count if window_count else 0.0

    hydrophobic_moment_value = hydrophobic_moment(sequence, window=11)
    amphipathic = 1.0 if hydrophobic_moment_value >= amphipathic_threshold else 0.0

    surface_hydrophobic: float | None = None
    if structure_stats:
        exposure = structure_stats.get("hydrophobic_exposure", {})
        if isinstance(exposure, dict) and exposure.get("exposed_ratio") is not None:
            surface_hydrophobic = float(exposure["exposed_ratio"])

    parts = [
        (
            0.34,
            linear_score(float(len(mhc2_epitopes)), worst=max(6.0, length * 0.06), best=0.0),
            Evidence(
                label="MHC-II 强结合表位数",
                value=len(mhc2_epitopes),
                rationale=(
                    "辅助 T 细胞表位是治疗性蛋白免疫原性的主要驱动因素；"
                    "表位数量越多，产生抗药抗体的风险越高。"
                ),
            ),
        ),
        (
            0.22,
            linear_score(strong_ratio, worst=0.25, best=0.0),
            Evidence(
                label="强结合窗口占比",
                value=round(strong_ratio, 4),
                rationale="以 9 聚体滑窗统计所有潜在表位密度，反映整体免疫原性倾向。",
            ),
        ),
        (
            0.16,
            linear_score(best_score, worst=0.92, best=0.55),
            Evidence(
                label="最强表位结合倾向分",
                value=round(best_score, 4),
                rationale="单个高亲和力表位即可主导免疫应答，因此峰值同样重要。",
            ),
        ),
        (
            0.14,
            linear_score(float(len(mhc1_binders)) / max(1, window_count), worst=0.20, best=0.0),
            Evidence(
                label="MHC-I 表位密度",
                value=round(len(mhc1_binders) / max(1, window_count), 4),
                rationale="MHC-I 表位驱动细胞毒性 T 细胞应答，对胞内递送类产品影响更大。",
            ),
        ),
        (
            0.08,
            linear_score(hydrophobic_moment_value, worst=0.55, best=0.15),
            Evidence(
                label="两亲性螺旋倾向（Eisenberg 疏水矩）",
                value=round(hydrophobic_moment_value, 4),
                rationale="能形成两亲性螺旋的片段更易被 HLA-II 有效呈递，是实验验证过的正相关特征。",
            ),
        ),
        (
            0.06,
            linear_score(surface_hydrophobic, worst=0.30, best=0.08)
            if surface_hydrophobic is not None
            else None,
            Evidence(
                label="表面疏水暴露比例",
                value=round(surface_hydrophobic, 4) if surface_hydrophobic is not None else "不可用",
                rationale="表面疏水斑块与 B 细胞表位（抗体直接识别）相关；无结构时为不可用项。",
            ),
        ),
    ]

    score, evidence = aggregate_weighted(parts)
    loci = [epitope.to_dict() for epitope in mhc2_epitopes + mhc1_binders]

    return Metric(
        key="immunogenicity",
        label="免疫原性风险",
        score=round(score, 1),
        algorithm=(
            "MHC-II 锚点偏好启发式（P1/P4/P6/P9）+ MHC-I 锚点（P2/P9）"
            " + Eisenberg 疏水矩两亲性筛查"
        ),
        rationale=(
            "分数越高表示免疫原性风险越低。"
            "注意：本项为相对排序级筛查，不等同于 NetMHCIIpan 的绝对亲和力预测；"
            "如需法规申报级数据，建议接入 IEDB 或商业预测器（接口已预留）。"
        ),
        confidence=0.4,
        evidence=evidence,
        locus=loci[:300],
        meta={
            "mhc2_epitope_count": len(mhc2_epitopes),
            "mhc1_top_binders": len(mhc1_binders),
            "strong_window_ratio": round(strong_ratio, 4),
            "disclaimer": "筛查级启发式，非绝对亲和力预测；建议关键候选送 NetMHCIIpan 复核。",
        },
    )
