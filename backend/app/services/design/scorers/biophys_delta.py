"""突变引起的理化变化与"新增风险位点"判定。

设计思路
--------
一个突变是否"安全"，很大程度取决于它**是否引入了新的化学不稳定位点**。
这类判断是纯规则、完全可复现的，因此作为评分卡里最硬的一条证据：

* 新增脱酰胺基序（Asn-Gly 等）—— 碱稳定性直接受损；
* 新增 Asp-Pro 酸敏感键 —— 酸稳定性受损；
* 新增易氧化残基（Met/Cys/Trp）；
* 新增游离半胱氨酸 / 破坏现有二硫键 —— 折叠与聚集风险；
* 在 α-螺旋中引入脯氨酸/甘氨酸 —— 二级结构破坏；
* 在**埋藏位点**做大幅体积变化或引入电荷 —— 堆积缺陷与去溶剂化惩罚。

所有判定都给出中文依据，实验人员可以直接核对。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ....core.config import load_platform_config
from ....core.logging import get_logger
from ...property.aggregation import HOTSPOT_THRESHOLD, aggregation_profile
from ...property.protease_sites import TOOL_RULES, find_cleavage_sites
from ...property.stability import DEAMIDATION_MOTIFS
from ...sequence.residue_properties import (
    charge_change,
    is_conservative,
    volume_change,
)
from ..scanner import PositionContext, mutation_label

logger = get_logger(__name__)

#: 局部窗口半径（检查基序时以突变位点为中心）
LOCAL_RADIUS = 4

#: 各类风险的扣分权重
PENALTIES: dict[str, float] = {
    "new_deamidation": 14.0,
    "new_acid_labile": 8.0,
    "new_oxidation": 12.0,
    "new_oxidation_minor": 2.5,
    "new_n_glycosylation": 6.0,
    "free_cysteine": 18.0,
    "broken_disulfide": 20.0,
    "pro_in_helix": 12.0,
    "gly_in_helix": 8.0,
    "new_aggregation_hotspot": 10.0,
    "buried_charge": 10.0,
    "new_cleavage_site": 8.0,
}


@dataclass
class RiskAssessment:
    """一个突变的化学风险评估。"""

    safety_score: float
    flags: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    deltas: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "safety_score": self.safety_score,
            "flags": self.flags,
            "details": self.details,
            "deltas": self.deltas,
        }


def _local_motifs(sequence: str, center: int, radius: int = LOCAL_RADIUS) -> set[str]:
    """提取突变位点邻域内的敏感基序集合。

    以"突变前 vs 突变后"的集合差判定"新增"，避免把本来就存在的老位点算作新风险。
    """
    start = max(0, center - radius)
    end = min(len(sequence), center + radius + 3)
    window = sequence[start:end]
    motifs: set[str] = set()

    for motif in DEAMIDATION_MOTIFS:
        offset = 0
        while True:
            index = window.find(motif, offset)
            if index < 0:
                break
            motifs.add(f"deamidation:{start + index}:{motif}")
            offset = index + 1
    for index in range(len(window) - 1):
        if window[index : index + 2] == "DP":
            motifs.add(f"acid_labile:{start + index}:DP")
    for index, char in enumerate(window):
        if char in "MCWHY":
            motifs.add(f"oxidation:{start + index}:{char}")
    for index in range(len(window) - 2):
        if window[index] == "N" and window[index + 1] != "P" and window[index + 2] in "ST":
            motifs.add(f"nglyc:{start + index}:{window[index:index + 3]}")

    # 工具酶识别位点（新增位点会导致标签切除时误切）
    for rule in TOOL_RULES:
        import re

        for match in re.finditer(rule.pattern, window):
            motifs.add(f"cleavage:{rule.key}:{start + match.start()}")

    return motifs


def _cys_neighbours(sequence: str, position: int, radius: int = 8) -> list[int]:
    """查找突变位点附近的半胱氨酸（用于判断能否形成二硫键）。"""
    start = max(0, position - radius)
    end = min(len(sequence), position + radius + 1)
    return [index for index in range(start, end) if index != position and sequence[index] == "C"]


def assess_mutation_risk(
    sequence: str,
    position: int,
    mutant: str,
    context: PositionContext | None = None,
    protein_type: str = "generic",
    *,
    wild_profile: list[float] | None = None,
) -> RiskAssessment:
    """评估单个突变的化学/结构风险。

    Args:
        wild_profile: 野生型的聚集倾向轨道。批量评估时**必须**由调用方预计算一次
            并传入——否则每条候选都要重算一遍全序列轨道（O(L)），
            在 L×19 条候选上会放大成不可接受的开销。
    """
    wild_type = sequence[position]
    mutant_sequence = sequence[:position] + mutant + sequence[position + 1 :]

    flags: list[str] = []
    details: dict[str, Any] = {}
    penalty = 0.0

    # ---------- 1) 新增化学不稳定基序 ----------
    before = _local_motifs(sequence, position)
    after = _local_motifs(mutant_sequence, position)
    new_motifs = after - before

    new_deamidation = sorted({item.split(":")[2] for item in new_motifs if item.startswith("deamidation:")})
    if new_deamidation:
        penalty += PENALTIES["new_deamidation"]
        flags.append(f"新增脱酰胺敏感基序：{'、'.join(new_deamidation)}")
        details["new_deamidation_motifs"] = new_deamidation

    if any(item.startswith("acid_labile:") for item in new_motifs):
        penalty += PENALTIES["new_acid_labile"]
        flags.append("新增 Asp-Pro 酸敏感肽键")

    new_oxidation = sorted({item.split(":")[-1] for item in new_motifs if item.startswith("oxidation:")})
    if new_oxidation:
        penalty += PENALTIES["new_oxidation"]
        flags.append(f"新增易氧化残基：{'、'.join(new_oxidation)}")
        details["new_oxidation_residues"] = new_oxidation

    if any(item.startswith("nglyc:") for item in new_motifs):
        penalty += PENALTIES["new_n_glycosylation"]
        flags.append("新增 N-糖基化序列子（N-X-S/T）")

    new_cleavage = sorted({item.split(":")[1] for item in new_motifs if item.startswith("cleavage:")})
    if new_cleavage:
        penalty += PENALTIES["new_cleavage_site"]
        flags.append(f"新增工具酶识别位点：{'、'.join(new_cleavage)}，可能影响标签切除")

    # ---------- 2) 半胱氨酸相关 ----------
    if mutant == "C" and wild_type != "C":
        neighbours = _cys_neighbours(sequence, position)
        if neighbours:
            flags.append(
                f"引入半胱氨酸，附近存在 Cys（位置 {[n + 1 for n in neighbours]}），"
                "可能形成非预期二硫键，需实验确认配对"
            )
            details["new_cysteine"] = {"position": position + 1, "nearby_cys": [n + 1 for n in neighbours]}
        else:
            penalty += PENALTIES["free_cysteine"]
            flags.append("引入游离半胱氨酸且附近无配对 Cys，易发生氧化与错配聚集")
            details["new_cysteine"] = {"position": position + 1, "nearby_cys": []}

    if wild_type == "C" and mutant != "C":
        penalty += PENALTIES["broken_disulfide"]
        flags.append("该位点为半胱氨酸，突变将破坏潜在二硫键，显著影响折叠")

    # ---------- 3) 二级结构破坏 ----------
    if context is not None:
        if context.is_helix and mutant == "P":
            penalty += PENALTIES["pro_in_helix"]
            flags.append(f"在 {context.describe()} 中引入脯氨酸，脯氨酸是经典螺旋破坏者")
        if context.is_helix and mutant == "G" and wild_type != "G":
            penalty += PENALTIES["gly_in_helix"]
            flags.append(f"在 {context.describe()} 中引入甘氨酸，提高主链柔性、降低螺旋稳定性")

        # ---------- 4) 埋藏位点的堆积与电荷 ----------
        risk_config = load_platform_config().get("design", {}).get("risk", {})
        coefficient = float(risk_config.get("volume_change_penalty", 60.0))

        if context.is_buried:
            dv = abs(volume_change(wild_type, mutant))
            if dv > 40:
                extra = min(30.0, dv / max(1.0, coefficient) * 10.0 + 8.0)
                penalty += extra
                flags.append(
                    f"埋藏位点（相对SASA={context.relative_sasa:.2f}）体积变化 {dv:.0f} Å³，"
                    "可能造成核心堆积缺陷"
                )
                details["volume_change"] = round(volume_change(wild_type, mutant), 1)

            dq = charge_change(wild_type, mutant)
            if abs(dq) > 0.9:
                penalty += PENALTIES["buried_charge"]
                flags.append("在埋藏位点引入电荷，需付出去溶剂化能量代价")

    # ---------- 5) 新增聚集热点 ----------
    # 注意：只重算位点邻域即可。全序列轨道在邻域之外不受该突变影响，
    # 但 aggregation_profile 是滑窗计算，邻域边缘会有 3 个残基的耦合，
    # 因此窗口取 [position-3-window, position+3+window] 才是严格等价的。
    if wild_profile is None:
        wild_profile = aggregation_profile(sequence, window=7)

    window = 7
    local_start = max(0, position - 3 - window)
    local_end = min(len(sequence), position + 4 + window)

    wild_local = wild_profile[local_start:local_end]
    # 对扩边后的子序列重算滑窗：直接切全长轨道会因序列变短导致窗口索引偏移，
    # 用等长的子序列重算即可保证与全长轨道逐位对齐。
    mutant_local = aggregation_profile(mutant_sequence[local_start:local_end], window=window)

    start = max(0, position - 3) - local_start
    end = min(len(sequence), position + 4) - local_start
    if mutant_local and wild_local:
        local_max = max(mutant_local[start:end])
        wild_max = max(wild_local[start:end])
        if local_max >= HOTSPOT_THRESHOLD and local_max > wild_max + 8:
            penalty += PENALTIES["new_aggregation_hotspot"]
            flags.append(
                f"在突变位点邻域新增聚集热点（倾向分 {local_max:.0f} > 阈值 {HOTSPOT_THRESHOLD:.0f}）"
            )
            details["local_aggregation"] = {"before": round(wild_max, 1), "after": round(local_max, 1)}

    safety = max(0.0, min(100.0, 100.0 - penalty))

    deltas = {
        "volume_change": round(volume_change(wild_type, mutant), 1),
        "charge_change": round(charge_change(wild_type, mutant), 2),
        "charge_change_ph7": round(charge_change(wild_type, mutant), 2),
        "is_conservative": is_conservative(wild_type, mutant),
        "label": mutation_label(sequence, position, mutant),
        "buried": context.is_buried if context else None,
        "secondary_structure": context.secondary_structure if context else None,
    }

    return RiskAssessment(
        safety_score=round(safety, 1),
        flags=flags,
        details=details,
        deltas=deltas,
    )
