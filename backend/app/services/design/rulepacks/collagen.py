"""胶原蛋白规则包。

领域知识
--------
胶原三股螺旋由三条左手螺旋链缠绕而成，其结构约束极其严格：

1. **Gly 必须出现在每三个残基的第一位**（Gly-X-Y）。Gly 是唯一足够小的残基，
   能把侧链塞进螺旋轴心；任何 Gly→X 的替换都会直接破坏三股螺旋，因此
   **Gly 位严格保护**。
2. **Y 位脯氨酸的 4-羟化**是热稳定性的决定性因素：羟脯氨酸通过水桥网络与
   主链羰基形成额外氢键，把 Tm 提高 10 °C 以上。因此 Y 位引入 Pro 是
   最直接的稳定性提升手段。
3. **X/Y 位的带电残基**可在三条链之间形成离子型相互作用（如 K–D / K–E），
   是另一条被广泛验证的稳定性提升路径。
4. **交联位点**：Y 位赖氨酸经赖氨酰氧化酶氧化后形成共价交联，是成熟胶原
   力学强度的来源；在 Y 位引入 Lys 可为交联工程提供位点。
5. 大侧链芳香族残基（F/W/Y）在 X/Y 位会与相邻链发生空间冲突，应劝退。

实现完全基于上述**三股螺旋周期性**判断（:func:`gly_xy_phase`），不依赖任何
绝对残基编号，因此对人类 COL1A1 全长、重组胶原片段、模型肽都适用。
"""

from __future__ import annotations

from typing import Any

from ....core.config import load_platform_config
from ....core.logging import get_logger
from ...sequence.feature_utils import (
    collagen_repeat_density,
    gly_xy_phase,
    is_gly_xy_repeat,
)
from ..scanner import PositionContext
from .base import RuleContext, RulePack, RuleVerdict

logger = get_logger(__name__)

#: 在 X/Y 位会与相邻链发生空间冲突的残基
BULKY_CONFLICT = frozenset("FWY")
#: 可在链间形成离子型相互作用的带电残基
INTERCHAIN_CHARGED = frozenset("DEKR")

#: 认定"三股螺旋结构域起点"所需的连续 Gly-X-Y 三联体数。
#: 取 20 是实测校准结果：在 COL1A1（P02452）上，阈值 10/15 会把起点判到 109 位
#: （N-前肽中存在零星的 Gly-X-Y 样片段），而 20 及以上稳定给出 **179**，
#: 与 UniProt 对该蛋白三股螺旋域起始位置的注释完全一致。
MIN_TRIPLETS_FOR_DOMAIN = 20


def helical_domain_onset(sequence: str, min_triplets: int = MIN_TRIPLETS_FOR_DOMAIN) -> int:
    """返回三股螺旋结构域的起点（0-based），无法判定时返回 0。

    判定方式：寻找首个包含 ``min_triplets`` 个连续处于 Gly-X-Y 相位的 Gly 的窗口。
    相比"第一个 Gly"要稳健得多——序列前段常有孤立的 Gly 残基。
    """
    positions = [
        index
        for index, residue in enumerate(sequence)
        if residue == "G" and gly_xy_phase(sequence, index) == 0
    ]
    if len(positions) < min_triplets:
        return 0

    # 连续（间距为 3）的 Gly 计数
    run = 1
    for index in range(1, len(positions)):
        if positions[index] - positions[index - 1] == 3:
            run += 1
            if run >= min_triplets:
                return positions[index - min_triplets + 1]
        else:
            run = 1
    return 0


class CollagenRulePack(RulePack):
    """胶原蛋白专用规则。"""

    category = "collagen"
    label = "胶原蛋白"
    description = "三股螺旋稳定性、羟脯氨酸位点优化与交联位点设计"

    def __init__(self, gly_strict: bool = True, protect_non_product_region: bool = True) -> None:
        self.gly_strict = gly_strict
        self._protect_non_product_region = protect_non_product_region

    # ---------------- 保护位点 ---------------- #
    def protected_positions(self, sequence: str) -> dict[int, str]:
        """保护三股螺旋结构域之前的区段，以及全部 Gly-X-Y 相位的 Gly。"""
        protected: dict[int, str] = {}

        # 1) 三股螺旋结构域之前的区段（信号肽 + N-前肽）在成熟过程中被切除。
        #    实测教训：全长 COL1A1 的 Top 候选曾落在第 2 位（信号肽内），
        #    对最终产品毫无意义。这里用"首个持续 Gly-X-Y 重复块"作为结构域起点，
        #    对重组胶原片段（本身就是螺旋域）不会误触发。
        if self._protect_non_product_region:
            onset = helical_domain_onset(sequence)
            if onset and onset > 0:
                for index in range(min(onset, len(sequence))):
                    protected[index] = (
                        f"位于三股螺旋结构域之前（第 1-{onset} 位，信号肽/N-前肽区），"
                        "成熟过程中被切除，该区段的突变推荐对最终产品无意义"
                    )

        # 2) 三股螺旋 Gly 位严格保护
        for index, residue in enumerate(sequence):
            if residue != "G":
                continue
            phase = gly_xy_phase(sequence, index)
            if phase == 0 or is_gly_xy_repeat(sequence, index):
                protected[index] = (
                    "三股螺旋 Gly 位：Gly 是唯一能把侧链塞入螺旋轴心的残基，"
                    "替换将直接破坏三股螺旋结构（Gly-X-Y 规则）"
                )
        return protected

    # ---------------- 活性/功能偏好 ---------------- #
    @staticmethod
    def _phase_context(sequence: str, position: int) -> tuple[int | None, str]:
        phase = gly_xy_phase(sequence, position)
        if phase is None:
            return None, "非 Gly-X-Y 重复区"
        return phase, {0: "Gly 位", 1: "X 位", 2: "Y 位"}[phase]

    def preferred_residues(self, sequence: str, position: int) -> set[str]:
        phase, _ = self._phase_context(sequence, position)
        if phase == 2:
            # Y 位：Pro（羟化位点）与 Lys（交联位点）优先
            return {"P", "K"}
        if phase == 1:
            # X 位：Pro 可稳定螺旋；带电残基可形成链间作用
            return {"P"} | INTERCHAIN_CHARGED
        if phase is None:
            # 非重复区：以带电残基做可溶性/稳定性优化
            return set(INTERCHAIN_CHARGED)
        return set()

    def discouraged_residues(self, sequence: str, position: int) -> set[str]:
        phase, _ = self._phase_context(sequence, position)
        if phase in (1, 2):
            # X/Y 位避免大侧链芳香族，避免引入 Gly 造成周期注册混乱
            return set(BULKY_CONFLICT) | {"G", "W"}
        return {"G"} if phase is None else set()

    def activity_factor(self, rule_context: RuleContext) -> RuleVerdict:
        sequence = rule_context.sequence
        position = rule_context.position
        mutant = rule_context.mutant
        phase, phase_label = self._phase_context(sequence, position)

        if phase == 0:
            return RuleVerdict(
                factor=0.0,
                note="该位点为三股螺旋 Gly 位，任何替换都会破坏 Gly-X-Y 规则",
                flags=["触及三股螺旋必需 Gly 位"],
                evidence={"phase": 0},
            )

        if phase == 2:
            if mutant == "P":
                return RuleVerdict(
                    factor=1.0,
                    note=(
                        "Y 位引入脯氨酸：可被脯氨酰-4-羟化酶转化为羟脯氨酸，"
                        "是提升三股螺旋热稳定性最直接的路径（文献报道 Tm 可提升 10 °C 以上）"
                    ),
                    flags=["新增羟脯氨酸候选位点"],
                    evidence={"phase": 2, "role": "hyp_candidate"},
                )
            if mutant == "K":
                return RuleVerdict(
                    factor=0.95,
                    note="Y 位引入赖氨酸：为赖氨酰氧化酶介导的共价交联提供位点，可增强力学强度",
                    flags=["新增交联候选位点"],
                    evidence={"phase": 2, "role": "crosslink_candidate"},
                )
            if mutant in INTERCHAIN_CHARGED:
                return RuleVerdict(
                    factor=0.85,
                    note="Y 位引入带电残基：可在三条链之间形成离子型相互作用，提升三股螺旋稳定性",
                    evidence={"phase": 2, "role": "interchain_ionic"},
                )
            if mutant in BULKY_CONFLICT:
                return RuleVerdict(
                    factor=0.18,
                    note="Y 位引入大侧链芳香族残基：侧链朝向螺旋外侧但体积过大，易与相邻链冲突",
                    evidence={"phase": 2, "role": "steric_conflict"},
                )
            return RuleVerdict(factor=0.45, note=f"Y 位替换为 {mutant}，对三股螺旋稳定性影响中性")

        if phase == 1:
            if mutant == "P":
                return RuleVerdict(
                    factor=0.80,
                    note="X 位引入脯氨酸：可提高主链刚性，间接增强三股螺旋稳定性（3-羟化较弱）",
                    evidence={"phase": 1, "role": "x_pro"},
                )
            if mutant in INTERCHAIN_CHARGED:
                return RuleVerdict(
                    factor=0.78,
                    note="X 位引入带电残基：参与链间离子型相互作用",
                    evidence={"phase": 1, "role": "interchain_ionic"},
                )
            if mutant in BULKY_CONFLICT:
                return RuleVerdict(
                    factor=0.25,
                    note="X 位引入大侧链芳香族残基，存在链间空间冲突风险",
                    evidence={"phase": 1, "role": "steric_conflict"},
                )
            return RuleVerdict(factor=0.5, note=f"X 位替换为 {mutant}，影响中性")

        # 非重复区（端肽/非螺旋区）
        if mutant in INTERCHAIN_CHARGED:
            return RuleVerdict(
                factor=0.7,
                note="该位点位于非 Gly-X-Y 重复区，引入带电残基主要改善可溶性",
                evidence={"phase": None},
            )
        return RuleVerdict(factor=0.45, note="该位点位于非重复区，对三股螺旋稳定性无直接贡献")

    # ---------------- 策略说明 ---------------- #
    def design_hints(self, sequence: str) -> list[str]:
        phase_counts = {0: 0, 1: 0, 2: 0, "none": 0}
        hydroxypotential = 0
        lysine_crosslink = 0
        for index, residue in enumerate(sequence):
            phase = gly_xy_phase(sequence, index)
            if phase is None:
                phase_counts["none"] += 1
                continue
            phase_counts[phase] += 1
            if phase == 2 and residue == "P":
                hydroxypotential += 1
            if phase == 2 and residue == "K":
                lysine_crosslink += 1

        density = collagen_repeat_density(sequence)
        hints = [
            f"Gly-X-Y 相位统计：Gly 位 {phase_counts[0]}、X 位 {phase_counts[1]}、Y 位 {phase_counts[2]}、"
            f"非重复区 {phase_counts['none']}；重复密度 {density:.2f}。",
            f"天然羟脯氨酸候选位点（Y 位 Pro）：{hydroxypotential} 个；"
            f"天然交联候选位点（Y 位 Lys）：{lysine_crosslink} 个。",
            "稳定性提升首选：把 Y 位替换为 Pro（新增羟化位点），其次在 X/Y 位引入 K/D/E（链间离子作用）。",
            "交联设计：在 Y 位引入 Lys，配合赖氨酰氧化酶处理形成共价交联。",
            "严禁改变 Gly 位——这会直接中断三股螺旋。",
        ]
        if density < 0.3:
            hints.append(
                "注意：本序列的 Gly-X-Y 重复密度低于 0.30，可能是非纤维状胶原或片段，"
                "三股螺旋相关建议的适用性需人工判断。"
            )
        return hints

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["protected_rule"] = "保护全部处于 Gly-X-Y 相位的 Gly"
        payload["key_motif"] = "Gly-X-Y（三股螺旋必需周期）"
        return payload
