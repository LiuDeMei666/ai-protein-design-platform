"""规则包基类。

规则包把**领域知识**从打分算法中解耦出来，让企业生物技术专家可以在不改动
核心代码的前提下扩充改造策略。每个规则包提供四件事：

1. :meth:`protected_positions` —— 绝对不能动的位点（催化残基、结合热点等）；
2. :meth:`activity_factor` —— 该替换是否符合本蛋白类型的活性改造偏好（0-1）；
3. :meth:`preferred_residues` / :meth:`discouraged_residues` —— 推荐/劝退的替换；
4. :meth:`design_hints` —— 面向研发人员的中文改造策略说明。

**所有保护位点都必须给出可核对的判定依据**，禁止硬编码魔数编号
（不同来源的序列编号差异极大，例如枯草杆菌蛋白酶前体与成熟酶的编号相差 100+）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ....core.config import load_platform_config
from ....core.logging import get_logger
from ..scanner import PositionContext

logger = get_logger(__name__)


def _rulepack_flag(category: str, key: str, default: bool) -> bool:
    """读取某个规则包的布尔配置项。"""
    config = load_platform_config().get("rulepacks", {}).get(category, {}) or {}
    return bool(config.get(key, default))


@dataclass
class RuleContext:
    """规则评估上下文。"""

    sequence: str
    position: int
    wild_type: str
    mutant: str
    context: PositionContext | None = None

    @property
    def label(self) -> str:
        return f"{self.wild_type}{self.position + 1}{self.mutant}"


@dataclass
class RuleVerdict:
    """规则给出的判断。"""

    factor: float = 0.5
    note: str = ""
    flags: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


class RulePack:
    """规则包基类。"""

    category: str = "generic"
    label: str = "通用蛋白"
    description: str = ""

    # ---------------- 保护位点 ---------------- #
    def protected_positions(self, sequence: str) -> dict[int, str]:
        """返回 ``{0-based 位点: 保护原因}``。"""
        return {}

    def n_terminal_cleavage_protection(self, sequence: str) -> dict[int, str]:
        """信号肽区保护（适用于所有蛋白类型）。

        分泌型重组蛋白的 N 端信号肽在成熟过程中被切除，在这些位点推荐突变
        对最终产品无效。实测中全长 COL1A1 的 Top 候选曾落在第 2 位、
        蛋白 A 落在第 5 位，都属于这一类问题。

        采用启发式识别（见 :func:`detect_signal_peptide`），可通过
        配置项 ``rulepacks.<type>.protect_signal_peptide`` 关闭。
        """
        from ...sequence.feature_utils import detect_signal_peptide

        span = detect_signal_peptide(sequence)
        if span is None:
            return {}
        _start, end = span
        reason = (
            f"推定信号肽区（第 1-{end} 位，启发式识别：N 端存在疏水核心），"
            "成熟过程中被切除，在该区段推荐突变对最终产品无意义"
        )
        return {index: reason for index in range(end)}

    def merged_protections(self, sequence: str) -> dict[int, str]:
        """合并本规则包的全部保护位点（含信号肽区）。

        调用方应使用本方法而不是直接调用 :meth:`protected_positions`。
        """
        merged: dict[int, str] = {}
        protect_signal = _rulepack_flag(self.category, "protect_signal_peptide", True)
        if protect_signal:
            merged.update(self.n_terminal_cleavage_protection(sequence))
        # 专用规则包的结论优先级更高：同一位点以更具体的原因覆盖
        for position, reason in self.protected_positions(sequence).items():
            merged[position] = reason
        return merged

    # ---------------- 活性偏好 ---------------- #
    def preferred_residues(self, sequence: str, position: int) -> set[str]:
        """该位点推荐的替换残基（不包含野生型自身）。"""
        return set()

    def discouraged_residues(self, sequence: str, position: int) -> set[str]:
        """该位点劝退的替换残基。"""
        return set()

    def activity_factor(self, rule_context: RuleContext) -> RuleVerdict:
        """该替换对"目标活性"的影响因子（0-1，越大越有利）。"""
        return RuleVerdict(factor=0.5, note="通用规则：无特定活性偏好")

    # ---------------- 策略说明 ---------------- #
    def design_hints(self, sequence: str) -> list[str]:
        """面向用户的中文改造策略说明。"""
        return []

    def describe(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "label": self.label,
            "description": self.description,
        }


class GenericRulePack(RulePack):
    """通用蛋白规则：所有蛋白类型的基础约束。"""

    category = "generic"
    label = "通用蛋白"
    description = "表面优先、避免新增游离半胱氨酸、避免在螺旋中引入脯氨酸"

    def __init__(self, avoid_free_cysteine: bool = True, avoid_glycine_in_helix: bool = True) -> None:
        self.avoid_free_cysteine = avoid_free_cysteine
        self.avoid_glycine_in_helix = avoid_glycine_in_helix

    def discouraged_residues(self, sequence: str, position: int) -> set[str]:
        discouraged: set[str] = set()
        if self.avoid_free_cysteine and sequence[position] != "C":
            # 只有附近已有可配对 Cys 时才允许引入新 Cys
            nearby = any(
                sequence[index] == "C"
                for index in range(max(0, position - 8), min(len(sequence), position + 9))
                if index != position
            )
            if not nearby:
                discouraged.add("C")
        return discouraged

    def activity_factor(self, rule_context: RuleContext) -> RuleVerdict:
        context = rule_context.context
        if context is None:
            return RuleVerdict(factor=0.5, note="无结构信息，采用中性因子")

        if context.is_buried:
            return RuleVerdict(
                factor=0.38,
                note=f"该位点埋藏（相对SASA={context.relative_sasa:.2f}），替换易破坏疏水核心",
                evidence={"buried": True},
            )
        if context.relative_sasa is not None and context.relative_sasa > 0.45:
            return RuleVerdict(
                factor=0.85,
                note=f"该位点高度暴露（相对SASA={context.relative_sasa:.2f}），是安全改造位点",
                evidence={"exposed": True},
            )
        return RuleVerdict(factor=0.6, note="该位点部分暴露，改造风险中等")

    def design_hints(self, sequence: str) -> list[str]:
        return [
            "优先选择相对 SASA > 0.45 的表面位点，改造空间最大且对核心无扰动。",
            "避免在 α-螺旋/β-折叠内部引入脯氨酸或甘氨酸。",
            "除非能与既有 Cys 形成二硫键，否则不要引入新的半胱氨酸。",
        ]
