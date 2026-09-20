"""翻译后修饰（PTM）敏感位点预测。

覆盖的修饰类型与依据
--------------------
==============  ============================================================
修饰            触发基序 / 条件（全部为已发表共识）
==============  ============================================================
脱酰胺          Asn-Gly > Asn-Ser/His/Thr/Asp；Gln 慢得多。碱性/pH 5 加速
异构化          Asp-Gly（琥珀酰亚胺中间体，天冬氨酸异构化经典热点）
氧化            Met（最易）、Cys、Trp、His、Tyr；Met 在 P1' 为芳香族时更敏感
N-糖基化        N-X-S/T 序列子，X ≠ P（真核表达体系）
O-糖基化        S/T 富集区（真核/酵母体系）
胶原羟化化      Gly-X-Y 中 Y 位的 Pro -> 4-羟脯氨酸（胶原三股螺旋稳定关键）
N 端焦谷氨酸    N 端为 Gln/Glu 时自发环化
糖化            Lys 侧链被还原糖修饰（重组蛋白制剂长期储存）
==============  ============================================================

每一项都给出位点级明细，前端在"位点轨道图"上按类型分色渲染。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ...core.logging import get_logger
from ..sequence.feature_utils import gly_xy_phase, is_gly_xy_repeat
from .metric import Evidence, Metric, aggregate_weighted, linear_score

logger = get_logger(__name__)

#: Asp 异构化敏感基序（琥珀酰亚胺形成速率）
ISOMERIZATION_MOTIFS: dict[str, float] = {
    "DG": 1.00, "DS": 0.72, "DT": 0.65, "DH": 0.60, "DD": 0.45, "DA": 0.35, "NG": 0.80,
}
#: N-糖基化序列子：N-X-S/T，X != P
N_GLYCOSYLATION = re.compile(r"N[^P](?=[ST])")
#: Met 氧化敏感性提升的后续残基（P1' 为芳香族/大侧链时更易被氧化）
MET_SENSITIVE_FOLLOWERS = frozenset("FWYHKM")


@dataclass
class PTMReport:
    """PTM 位点报告。"""

    deamidation: list[dict[str, Any]] = field(default_factory=list)
    isomerization: list[dict[str, Any]] = field(default_factory=list)
    oxidation: list[dict[str, Any]] = field(default_factory=list)
    n_glycosylation: list[dict[str, Any]] = field(default_factory=list)
    o_glycosylation: list[dict[str, Any]] = field(default_factory=list)
    hydroxylation: list[dict[str, Any]] = field(default_factory=list)
    pyroglutamate: list[dict[str, Any]] = field(default_factory=list)
    glycation: list[dict[str, Any]] = field(default_factory=list)

    @property
    def all_loci(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for items in (
            self.deamidation,
            self.isomerization,
            self.oxidation,
            self.n_glycosylation,
            self.o_glycosylation,
            self.hydroxylation,
            self.pyroglutamate,
            self.glycation,
        ):
            payload.extend(items)
        return payload

    def counts(self) -> dict[str, int]:
        return {
            "deamidation": len(self.deamidation),
            "isomerization": len(self.isomerization),
            "oxidation": len(self.oxidation),
            "n_glycosylation": len(self.n_glycosylation),
            "o_glycosylation": len(self.o_glycosylation),
            "hydroxylation": len(self.hydroxylation),
            "pyroglutamate": len(self.pyroglutamate),
            "glycation": len(self.glycation),
        }


def _locus(position: int, residue: str, motif: str, kind: str, severity: float, note: str) -> dict[str, Any]:
    return {
        "position": position,
        "residue": residue,
        "motif": motif,
        "type": kind,
        "severity": round(severity, 2),
        "note": note,
    }


def analyze_ptm_sites(sequence: str, protein_type: str = "generic") -> PTMReport:
    """全量扫描各类 PTM 敏感位点。"""
    report = PTMReport()
    length = len(sequence)

    # --- 脱酰胺 ---
    for motif, weight in ISOMERIZATION_MOTIFS.items():
        if not motif.startswith("N"):
            continue
        for match in re.finditer(f"(?={motif})", sequence):
            report.deamidation.append(
                _locus(
                    match.start(),
                    "N",
                    motif,
                    "deamidation",
                    weight,
                    f"Asn-{motif[1]} 基序，碱性或弱酸条件下易脱酰胺生成 Asp/异天冬氨酸。",
                )
            )

    # --- 异构化（Asp 为主）---
    for motif, weight in ISOMERIZATION_MOTIFS.items():
        if not motif.startswith("D"):
            continue
        for match in re.finditer(f"(?={motif})", sequence):
            report.isomerization.append(
                _locus(
                    match.start(),
                    "D",
                    motif,
                    "isomerization",
                    weight,
                    f"Asp-{motif[1]} 基序，经琥珀酰亚胺中间体发生异构化，产生异天冬氨酸。",
                )
            )

    # --- 氧化 ---
    oxidation_priority = {"M": 1.0, "C": 0.85, "W": 0.6, "H": 0.5, "Y": 0.35}
    for index, char in enumerate(sequence):
        weight = oxidation_priority.get(char)
        if weight is None:
            continue
        if char == "M" and index + 1 < length and sequence[index + 1] in MET_SENSITIVE_FOLLOWERS:
            weight = min(1.0, weight + 0.15)
        if char == "C" and index > 0 and index + 1 < length:
            # 有配对 Cys 时氧化风险取决于是否形成二硫键，此处只标注
            weight = 0.6
        report.oxidation.append(
            _locus(
                index,
                char,
                char,
                "oxidation",
                weight,
                {
                    "M": "甲硫氨酸最易被氧化的残基，纯化与储存中需控制溶氧与金属离子。",
                    "C": "半胱氨酸易氧化为次磺酸/磺酸，或参与二硫键错配。",
                    "W": "色氨酸氧化会破坏结构并改变紫外吸收。",
                    "H": "组氨酸在金属催化氧化下易受损。",
                    "Y": "酪氨酸氧化生成二酪氨酸交联。",
                }[char],
            )
        )

    # --- N-糖基化序列子 ---
    for match in N_GLYCOSYLATION.finditer(sequence):
        report.n_glycosylation.append(
            _locus(
                match.start(),
                "N",
                sequence[match.start() : match.start() + 3],
                "n_glycosylation",
                0.9,
                "N-X-S/T 序列子（X≠P），真核表达体系中会被糖基化。",
            )
        )

    # --- O-糖基化（S/T 富集区）---
    for match in re.finditer(r"[ST]{2,}", sequence):
        span = match.group()
        if len(span) >= 3:
            report.o_glycosylation.append(
                _locus(
                    match.start(),
                    span[0],
                    span,
                    "o_glycosylation",
                    min(1.0, len(span) / 5.0),
                    "连续 Ser/Thr 富集区，酵母/哺乳动物体系中易发生 O-糖基化。",
                )
            )

    # --- 胶原羟脯氨酸位点 ---
    if protein_type == "collagen":
        for index, char in enumerate(sequence):
            if char != "P":
                continue
            phase = gly_xy_phase(sequence, index)
            # Y 位（相位 2）的 Pro 是 4-羟脯氨酸的主要位点
            if phase == 2:
                report.hydroxylation.append(
                    _locus(
                        index,
                        "P",
                        "G-X-P",
                        "hydroxylation",
                        1.0,
                        "胶原 Gly-X-Y 中 Y 位脯氨酸，是脯氨酰-4-羟化酶的主要底物，"
                        "羟化后显著增强三股螺旋热稳定性。",
                    )
                )
            elif phase == 1:
                report.hydroxylation.append(
                    _locus(
                        index,
                        "P",
                        "G-P-X",
                        "hydroxylation",
                        0.4,
                        "X 位脯氨酸，可被 3-羟化（较弱），对三股螺旋稳定贡献有限。",
                    )
                )

    # --- N 端焦谷氨酸 ---
    if sequence[:1] in ("Q", "E"):
        report.pyroglutamate.append(
            _locus(
                0,
                sequence[0],
                sequence[0],
                "pyroglutamate",
                0.8,
                "N 端 Gln/Glu 自发环化为焦谷氨酸，导致 N 端序列不均一与电荷变化。",
            )
        )

    # --- 赖氨酸糖化 ---
    for index, char in enumerate(sequence):
        if char == "K":
            report.glycation.append(
                _locus(
                    index,
                    "K",
                    "K",
                    "glycation",
                    0.3,
                    "赖氨酸侧链氨基可与还原糖发生糖化反应，影响长期制剂稳定性。",
                )
            )

    return report


def assess_ptm_risk(
    sequence: str, report: PTMReport | None = None, protein_type: str = "generic"
) -> Metric:
    """PTM 风险评分：分数越高表示修饰异质性风险越低。"""
    report = report or analyze_ptm_sites(sequence, protein_type)
    length = max(1, len(sequence))
    counts = report.counts()

    per_100 = lambda value: value / length * 100  # noqa: E731

    critical_deamidation = sum(
        1 for item in report.deamidation if float(item["severity"]) >= 0.7
    )
    critical_oxidation = sum(1 for item in report.oxidation if float(item["severity"]) >= 0.85)

    parts = [
        (
            0.26,
            linear_score(per_100(len(report.deamidation)), worst=3.0, best=0.0),
            Evidence(
                label="脱酰胺位点密度（每 100 残基）",
                value=round(per_100(len(report.deamidation)), 3),
                rationale="脱酰胺是重组蛋白最常见的化学降解途径，直接造成电荷异质性与活性下降。",
            ),
        ),
        (
            0.18,
            linear_score(per_100(len(report.isomerization)), worst=2.5, best=0.0),
            Evidence(
                label="异构化位点密度（每 100 残基）",
                value=round(per_100(len(report.isomerization)), 3),
                rationale="Asp-Gly 等基序经琥珀酰亚胺中间体异构化，产生难以分离的异天冬氨酸变体。",
            ),
        ),
        (
            0.22,
            linear_score(per_100(critical_oxidation), worst=2.5, best=0.0),
            Evidence(
                label="高敏感氧化位点密度（Met/Cys，每 100 残基）",
                value=round(per_100(critical_oxidation), 3),
                rationale="Met/Cys 氧化是最主要的氧化降解途径，可通过工艺控制溶氧与螯合金属离子缓解。",
            ),
        ),
        (
            0.14,
            linear_score(per_100(len(report.n_glycosylation)), worst=1.5, best=0.0),
            Evidence(
                label="N-糖基化序列子密度",
                value=round(per_100(len(report.n_glycosylation)), 3),
                rationale="真核体系中 N-X-S/T 会被糖基化；若产品需无糖基化则须通过突变消除。",
            ),
        ),
        (
            0.10,
            linear_score(float(len(report.pyroglutamate)), worst=1.0, best=0.0),
            Evidence(
                label="N 端焦谷氨酸风险",
                value=len(report.pyroglutamate),
                rationale="N 端 Gln/Glu 自发环化会导致 N 端不均一，影响电荷分布与活性。",
            ),
        ),
        (
            0.10,
            linear_score(float(critical_deamidation), worst=6.0, best=0.0),
            Evidence(
                label="高敏感脱酰胺位点数（Asn-Gly 类）",
                value=critical_deamidation,
                rationale="Asn-Gly / Asn-Ser 等高敏感位点的改造收益最高，是突变设计的优先靶点。",
            ),
        ),
    ]

    score, evidence = aggregate_weighted(parts)

    return Metric(
        key="ptm_sites",
        label="修饰位点风险",
        score=round(score, 1),
        algorithm="已发表共识基序扫描（脱酰胺/异构化/氧化/糖基化/羟化/焦谷氨酸/糖化）",
        rationale=(
            "分数越高表示翻译后修饰异质性风险越低。"
            "高敏感位点（尤其 Asn-Gly 与暴露 Met）是提升耐受性改造中最直接的靶点。"
        ),
        confidence=0.65,
        evidence=evidence,
        locus=report.all_loci[:600],
        meta={"counts": counts},
    )
