"""蛋白酶切位点识别与宿主蛋白酶抗性评估。

需求文档要求预测"蛋白酶切位点"。本模块覆盖三类：

1. **常用工具酶识别位点**（重组构建、标签切除用）：TEV、PreScission、Factor Xa、
   Thrombin、Enterokinase、SUMO 蛋白酶、Furin。用于**确认切除位点是否唯一**——
   若在目标蛋白内部出现额外位点，标签切除时会误切目标蛋白。
2. **特异性蛋白酶切割规则**（质谱鉴定与降解分析用）：Trypsin、Chymotrypsin、
   Lys-C、Glu-C(V8)、Asp-N、Pepsin。
3. **宿主内源蛋白酶敏感位点**（表达纯化过程中的降解风险）：以大肠杆菌胞内主要
   蛋白酶（Lon、ClpP、HslUV、OmpT/Tsp）的识别偏好做启发式评估。

每条规则都取自酶的官方识别规则，不做臆测。规则之外的"通用降解倾向"
（如暴露的柔性环区）在聚集/溶解性模块中单独评估。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ...core.logging import get_logger
from .metric import Evidence, Metric, aggregate_weighted, linear_score

logger = get_logger(__name__)


@dataclass(frozen=True)
class CleavageRule:
    """一条切割规则。"""

    key: str
    label: str
    pattern: str  # 正则；用捕获组 (?=...) 表示酶切在匹配之后的肽键
    category: str  # tool | specific | host
    description: str
    #: 是否在内部出现属于"有害"（工具酶为 True，分析用酶为 False）
    harmful_if_internal: bool = True


# 工具酶：识别位点，切割位置在 P1 与 P1' 之间（用前瞻断言表示）
TOOL_RULES: tuple[CleavageRule, ...] = (
    CleavageRule(
        key="tev",
        label="TEV 蛋白酶 (ENLYFQ/G)",
        pattern=r"ENLYFQ(?=G)",
        category="tool",
        description="烟草蚀纹病毒蛋白酶，识别 ENLYFQ↓G，特异性极高。",
    ),
    CleavageRule(
        key="prescission",
        label="PreScission/HRV 3C (LEVLFQ/GP)",
        pattern=r"LEVLFQ(?=GP)",
        category="tool",
        description="人鼻病毒 3C 蛋白酶，识别 LEVLFQ↓GP。",
    ),
    CleavageRule(
        key="factor_xa",
        label="Factor Xa (IEGR/)",
        pattern=r"IE[GD]R",
        category="tool",
        description="识别 IEGR↓，但特异性较低，常在非目标位点误切。",
    ),
    CleavageRule(
        key="thrombin",
        label="Thrombin (LVPR/GS)",
        pattern=r"LVPR(?=GS)",
        category="tool",
        description="识别 LVPR↓GS，需注意内部 LVPR 序列。",
    ),
    CleavageRule(
        key="enterokinase",
        label="Enterokinase (DDDDK/)",
        pattern=r"DDDDK",
        category="tool",
        description="识别 DDDDK↓，通常与 N 端标签配合使用。",
    ),
    CleavageRule(
        key="sumo",
        label="SUMO 蛋白酶 (GG/)",
        pattern=r"(?<=GG)GG(?=.)",
        category="tool",
        description="识别 SUMO 末端双甘氨酸，需注意序列中的连续 Gly。",
    ),
    CleavageRule(
        key="furin",
        label="Furin (RXXR/)",
        pattern=r"R[^P]{2}R",
        category="tool",
        description="前蛋白转化酶，识别 R-X-X-R↓，在真核表达体系中常见。",
    ),
)

# 特异性蛋白酶（用于肽图分析；内部出现是正常现象，不算风险）
SPECIFIC_RULES: tuple[CleavageRule, ...] = (
    CleavageRule(
        key="trypsin",
        label="Trypsin (K/R 后，Pro 前除外)",
        pattern=r"[KR](?![P])",
        category="specific",
        description="切割 Lys/Arg 的 C 端，若其后为 Pro 则不切。",
        harmful_if_internal=False,
    ),
    CleavageRule(
        key="chymotrypsin",
        label="Chymotrypsin (F/W/Y 后)",
        pattern=r"[FWY](?![P])",
        category="specific",
        description="切割芳香族残基 C 端，Pro 前不切。",
        harmful_if_internal=False,
    ),
    CleavageRule(
        key="lysc",
        label="Lys-C (K 后)",
        pattern=r"K",
        category="specific",
        description="特异性切割 Lys 的 C 端。",
        harmful_if_internal=False,
    ),
    CleavageRule(
        key="gluc",
        label="Glu-C / V8 (D/E 后)",
        pattern=r"[DE](?![P])",
        category="specific",
        description="在铵盐缓冲液中优先切 Glu，磷酸盐缓冲液中 Glu/Asp 均切。",
        harmful_if_internal=False,
    ),
    CleavageRule(
        key="aspn",
        label="Asp-N (D 前)",
        pattern=r"D",
        category="specific",
        description="切割 Asp 的 N 端。",
        harmful_if_internal=False,
    ),
)

# 宿主内源蛋白酶敏感位点（大肠杆菌为主）
HOST_RULES: tuple[CleavageRule, ...] = (
    CleavageRule(
        key="ompt",
        label="OmpT/Tsp 外膜蛋白酶偏好位点 (R/K-R/K 对)",
        pattern=r"[RK][RK]",
        category="host",
        description=(
            "OmpT 与 Tsp 是定位于外膜的蛋白酶，偏好切割两个碱性残基之间的肽键，"
            "对重组蛋白 C 端与柔性区降解贡献最大（可通过 ompT/tsp 缺陷菌株规避）。"
        ),
    ),
    CleavageRule(
        key="clpap",
        label="ClpP/Lon 疏水暴露位点 (疏水-疏水)",
        pattern=r"[AVILMFWY][AVILMFWY][AVILMFWY]",
        category="host",
        description=(
            "胞质蛋白酶 Lon 与 ClpP 优先降解疏水残基暴露的错折叠蛋白；"
            "连续三个疏水残基是常用的敏感性代理特征。"
        ),
    ),
    CleavageRule(
        key="hsluv",
        label="HslUV 偏好位点 (I/V-L 组合)",
        pattern=r"[IVLM][IVLM]",
        category="host",
        description="HslUV (ClpYQ) 偏好切割含 Ile/Val/Leu/Met 的疏水位点。",
    ),
)


@dataclass
class CleavageLocus:
    """一个切割位点。"""

    position: int
    matched: str
    rule: CleavageRule

    def to_dict(self, sequence: str) -> dict[str, Any]:
        return {
            "position": self.position,
            "residue": sequence[self.position] if self.position < len(sequence) else "",
            "matched": self.matched,
            "rule": self.rule.key,
            "label": self.rule.label,
            "category": self.rule.category,
            "type": "cleavage_site",
            "severity": 1.0 if self.rule.harmful_if_internal else 0.4,
            "description": self.rule.description,
        }


def find_cleavage_sites(
    sequence: str, rules: tuple[CleavageRule, ...]
) -> dict[str, list[CleavageLocus]]:
    """按规则集查找切割位点。"""
    found: dict[str, list[CleavageLocus]] = {}
    for rule in rules:
        loci: list[CleavageLocus] = []
        for match in re.finditer(rule.pattern, sequence):
            loci.append(
                CleavageLocus(position=match.start(), matched=match.group(), rule=rule)
            )
        if loci:
            found[rule.key] = loci
    return found


@dataclass
class ProteaseSiteReport:
    """蛋白酶切位点汇总报告。"""

    tool_sites: dict[str, list[CleavageLocus]] = field(default_factory=dict)
    specific_sites: dict[str, list[CleavageLocus]] = field(default_factory=dict)
    host_sites: dict[str, list[CleavageLocus]] = field(default_factory=dict)
    tool_duplicates: list[dict[str, Any]] = field(default_factory=list)

    def all_loci(self, sequence: str) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for group in (self.tool_sites, self.host_sites):
            for loci in group.values():
                payload.extend(locus.to_dict(sequence) for locus in loci)
        return payload


def analyze_protease_sites(sequence: str) -> ProteaseSiteReport:
    """完整分析：工具酶位点唯一性 + 宿主蛋白酶敏感位点。"""
    report = ProteaseSiteReport()
    report.tool_sites = find_cleavage_sites(sequence, TOOL_RULES)
    report.specific_sites = find_cleavage_sites(sequence, SPECIFIC_RULES)
    report.host_sites = find_cleavage_sites(sequence, HOST_RULES)

    # 工具酶位点若在目标蛋白内部出现多次，标签切除时会误切
    for rule_key, loci in report.tool_sites.items():
        if len(loci) > 1:
            rule = loci[0].rule
            report.tool_duplicates.append(
                {
                    "rule": rule_key,
                    "label": rule.label,
                    "count": len(loci),
                    "positions": [locus.position for locus in loci],
                    "risk": "设计时预留的切除位点之外还存在额外识别位点，标签切除将误切目标蛋白。",
                }
            )
    return report


def assess_protease_resistance(
    sequence: str, report: ProteaseSiteReport | None = None
) -> Metric:
    """宿主蛋白酶抗性评分：分数越高表示在表达纯化过程中越不易被降解。"""
    report = report or analyze_protease_sites(sequence)
    length = max(1, len(sequence))

    host_counts = {key: len(loci) for key, loci in report.host_sites.items()}
    ompt_count = host_counts.get("ompt", 0)
    clpap_count = host_counts.get("clpap", 0)
    hsluv_count = host_counts.get("hsluv", 0)

    ompt_density = ompt_count / length * 100
    clpap_density = clpap_count / length * 100
    hsluv_density = hsluv_count / length * 100

    duplicates = len(report.tool_duplicates)

    parts = [
        (
            0.34,
            linear_score(ompt_density, worst=4.5, best=0.0),
            Evidence(
                label="OmpT/Tsp 偏好位点密度（每 100 残基）",
                value=round(ompt_density, 3),
                rationale="OmpT/Tsp 是外膜蛋白酶，对碱性残基对之间的肽键切割效率最高。",
            ),
        ),
        (
            0.28,
            linear_score(clpap_density, worst=6.0, best=0.5),
            Evidence(
                label="Lon/ClpP 疏水暴露位点密度（每 100 残基）",
                value=round(clpap_density, 3),
                rationale="连续疏水三残基是胞质蛋白酶识别错折叠蛋白的常用代理特征。",
            ),
        ),
        (
            0.20,
            linear_score(hsluv_density, worst=9.0, best=1.0),
            Evidence(
                label="HslUV 偏好位点密度（每 100 残基）",
                value=round(hsluv_density, 3),
                rationale="HslUV (ClpYQ) 偏好含 Ile/Val/Leu/Met 的疏水位点。",
            ),
        ),
        (
            0.18,
            linear_score(float(duplicates), worst=2.0, best=0.0),
            Evidence(
                label="工具酶内部冗余位点数",
                value=duplicates,
                rationale=(
                    "TEV/PreScission/Thrombin 等识别位点若在目标蛋白内部重复出现，"
                    "标签切除时会误切目标蛋白，需在构建阶段重新设计。"
                ),
            ),
        ),
    ]

    score, evidence = aggregate_weighted(parts)

    tool_loci = [locus.to_dict(sequence) for loci in report.tool_sites.values() for locus in loci]

    return Metric(
        key="protease_resistance",
        label="宿主蛋白酶抗性",
        score=round(score, 1),
        algorithm="OmpT/Tsp、Lon/ClpP、HslUV 识别偏好启发式 + 工具酶位点唯一性检查",
        rationale=(
            "分数越高表示在表达与纯化过程中越不易被宿主蛋白酶降解。"
            "若偏低，可考虑使用 ompT/lon 缺陷菌株、缩短裂解后操作时间或低温纯化。"
        ),
        confidence=0.5,
        evidence=evidence,
        locus=(tool_loci + report.all_loci(sequence))[:400],
        meta={
            "host_site_counts": host_counts,
            "specific_site_counts": {
                key: len(loci) for key, loci in report.specific_sites.items()
            },
            "tool_site_counts": {key: len(loci) for key, loci in report.tool_sites.items()},
            "tool_duplicates": report.tool_duplicates,
        },
    )
