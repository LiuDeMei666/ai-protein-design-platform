"""蛋白 A（Staphylococcal protein A）规则包。

领域知识
--------
1. **结构组成**：蛋白 A 的 IgG 结合能力来自 5 个高度同源的串联 Ig 结合结构域
   （E、D、A、B、C，各约 58 残基）。这些结构域是**串联重复**，因此可以通过
   重复检测自动定位，再保护**跨重复保守**的残基（框架 + Fc 结合界面），
   这比硬编码编号稳健得多——不同菌株与构建体的编号差异很大。
2. **耐碱性改造**（需求文档明确要求）：碱性条件下（如 IgG 亲和层析的
   0.1 M NaOH 清洗）失活的主因是 **Asn 脱酰胺**，尤其是 Asn-Gly 基序。
   业界成熟的碱稳定改造策略就是把这些 Asn 替换为 Gln/Asp/Thr/Ser——
   既消除脱酰胺位点，又保持氢键能力。本规则包据此给替换方向打分。
3. **亲和力改造**：Fc 结合界面位于螺旋 I/II 的疏水面，涉及若干保守的
   Phe/Tyr/Leu/Gln 残基；这些位点在跨重复保守分析中会被自动保护。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ....core.config import load_platform_config
from ....core.logging import get_logger
from ...sequence.feature_utils import composition_counts
from .base import RuleContext, RulePack, RuleVerdict

logger = get_logger(__name__)


@dataclass
class RepeatUnit:
    """一个串联重复单元。"""

    start: int
    end: int
    identity: float

    @property
    def length(self) -> int:
        return self.end - self.start


def detect_tandem_repeats(
    sequence: str,
    min_domain: int = 40,
    max_domain: int = 75,
    min_copies: int = 2,
    min_identity: int = 30,
) -> tuple[list[RepeatUnit], float]:
    """检测串联重复结构域。

    做法：枚举结构域长度与相位偏移，把序列切成等长片段，计算**片段间平均一致性**，
    取一致性最高且片段数达到要求的方案。

    Returns:
        ``(重复单元列表, 平均一致性)``；未检出时返回 ``([], 0.0)``。
    """
    length = len(sequence)
    if length < min_domain * min_copies:
        return [], 0.0

    best: tuple[float, int, int, list[RepeatUnit]] | None = None

    for domain in range(min_domain, max_domain + 1):
        max_offset = min(domain, max(0, length - domain * min_copies))
        for offset in range(0, max_offset + 1):
            count = (length - offset) // domain
            if count < min_copies:
                continue
            segments = [
                sequence[offset + index * domain : offset + (index + 1) * domain]
                for index in range(count)
            ]
            pairs = 0
            total_identity = 0.0
            for i in range(count):
                for j in range(i + 1, count):
                    identical = sum(1 for a, b in zip(segments[i], segments[j]) if a == b)
                    total_identity += identical / domain
                    pairs += 1
            if pairs == 0:
                continue
            average = total_identity / pairs
            if average * 100 < min_identity:
                continue
            if best is None or average > best[0]:
                best = (average, domain, offset, [])

    if best is None:
        return [], 0.0

    average, domain, offset, _ = best
    count = (length - offset) // domain
    units = [
        RepeatUnit(
            start=offset + index * domain,
            end=offset + (index + 1) * domain,
            identity=round(average, 4),
        )
        for index in range(count)
    ]
    return units, round(average, 4)


def conserved_positions_across_repeats(
    sequence: str, units: list[RepeatUnit], threshold: float = 0.8
) -> dict[int, str]:
    """找出跨重复单元保守的位点（框架/功能核心）。"""
    if len(units) < 2:
        return {}

    domain = units[0].length
    conserved: dict[int, str] = {}

    for offset in range(domain):
        residues: list[str] = []
        for unit in units:
            index = unit.start + offset
            if index < unit.end and index < len(sequence):
                residues.append(sequence[index])
        if len(residues) < 2:
            continue
        counts: dict[str, int] = {}
        for residue in residues:
            counts[residue] = counts.get(residue, 0) + 1
        residue, count = max(counts.items(), key=lambda item: item[1])
        ratio = count / len(residues)
        if ratio >= threshold:
            for unit in units:
                index = unit.start + offset
                if index < unit.end and index < len(sequence):
                    conserved[index] = (
                        f"跨 {len(units)} 个 Ig 结合重复单元保守（{ratio:.0%} 一致，残基 {residue}），"
                        "属于框架/Fc 结合核心，改造风险高"
                    )
    return conserved


class ProteinARulePack(RulePack):
    """蛋白 A 专用规则。"""

    category = "protein_a"
    label = "蛋白 A"
    description = "IgG 亲和力保持与耐碱性提升（脱酰胺位点消除）"

    def __init__(self) -> None:
        config = load_platform_config().get("rulepacks", {}).get("protein_a", {})
        self.alkali_preferred = set(config.get("alkali_preferred", ["Q", "D", "E", "T", "S", "A"]))
        self._cache: dict[str, tuple[list[RepeatUnit], float, dict[int, str]]] = {}

    # ---------------- 重复结构分析 ---------------- #
    def _analyze(self, sequence: str) -> tuple[list[RepeatUnit], float, dict[int, str]]:
        if sequence not in self._cache:
            units, identity = detect_tandem_repeats(sequence)
            conserved = conserved_positions_across_repeats(sequence, units)
            self._cache[sequence] = (units, identity, conserved)
            if not units:
                logger.warning("未检测到串联重复结构域，蛋白 A 的保守位点保护未生效")
        return self._cache[sequence]

    def protected_positions(self, sequence: str) -> dict[int, str]:
        """保护跨重复保守位点，以及首个 Ig 结合结构域之前的非产品区段。

        为什么保护结构域之前的区段：蛋白 A 的产品形态是 Ig 结合结构域
        （通常 5 个，或工程化的 Z 结构域多聚体）。首个结构域之前的序列包含
        信号肽与前导区，在分泌与构建过程中不进入最终产品。
        实测教训：全长蛋白 A 的 Top 候选曾落在第 5 位（信号肽内），
        对产品毫无意义；而启发式信号肽识别对蛋白 A 这种 h-region 疏水性
        不典型的革兰氏阳性菌信号肽会漏判，因此这里改用**结构域边界**来判定，
        这是本蛋白更可靠的依据。
        """
        protected: dict[int, str] = {}

        units, _, conserved = self._analyze(sequence)
        if units and units[0].start > 0:
            for index in range(min(units[0].start, len(sequence))):
                protected[index] = (
                    f"位于首个 Ig 结合结构域之前（第 1-{units[0].start} 位，"
                    "信号肽/前导区），不进入最终产品，该区段的突变推荐无意义"
                )

        protected.update(conserved)
        return protected

    # ---------------- 活性/功能偏好 ---------------- #
    def preferred_residues(self, sequence: str, position: int) -> set[str]:
        """耐碱改造偏好：N/Q -> Q/D/E/T/S/A。"""
        wild_type = sequence[position]
        if wild_type in ("N", "Q"):
            return set(self.alkali_preferred) - {wild_type}
        return set()

    def discouraged_residues(self, sequence: str, position: int) -> set[str]:
        """引入新的 Asn/Gln 会带来脱酰胺风险。"""
        wild_type = sequence[position]
        discouraged: set[str] = set()
        if wild_type not in ("N", "Q"):
            discouraged.update({"N", "Q"})
        return discouraged

    def activity_factor(self, rule_context: RuleContext) -> RuleVerdict:
        sequence = rule_context.sequence
        position = rule_context.position
        wild_type = rule_context.wild_type
        mutant = rule_context.mutant
        units, identity, conserved = self._analyze(sequence)

        if position in conserved:
            if wild_type == "N" and mutant in self.alkali_preferred:
                return RuleVerdict(
                    factor=0.55,
                    note=(
                        f"该位点为跨重复保守的 Asn（脱酰胺热点），"
                        f"替换为 {mutant} 是耐碱改造的标准做法，但可能影响亲和力，需实测 KD"
                    ),
                    flags=["保守位点改造，需实测亲和力"],
                    evidence={"conserved": True, "alkali_engineering": True},
                )
            return RuleVerdict(
                factor=0.08,
                note=f"该位点保守：{conserved[position]}",
                flags=["触及保守功能位点"],
                evidence={"conserved": True},
            )

        # 非保守位点的耐碱改造
        if wild_type == "N":
            if mutant == "G":
                return RuleVerdict(
                    factor=0.05,
                    note="在 Asn 后引入 Gly 会形成 Asn-Gly 基序——这是脱酰胺速率最高的组合，必须避免",
                    flags=["新增 Asn-Gly 高敏感脱酰胺基序"],
                )
            if mutant in self.alkali_preferred:
                return RuleVerdict(
                    factor=0.92,
                    note=(
                        f"Asn -> {mutant}：消除脱酰胺位点，是提升耐碱性的首选改造"
                        "（Gln/Asp 保留氢键能力，Thr/Ser/Ala 体积相近）"
                    ),
                    flags=["耐碱改造"],
                    evidence={"alkali_engineering": True},
                )
            return RuleVerdict(factor=0.5, note=f"Asn -> {mutant}，不改变脱酰胺风险，也未改善")

        if wild_type == "Q" and mutant in self.alkali_preferred:
            return RuleVerdict(
                factor=0.80,
                note=f"Gln -> {mutant}：消除（较慢的）脱酰胺位点，进一步提升耐碱性",
                flags=["耐碱改造"],
                evidence={"alkali_engineering": True},
            )

        if mutant in ("N", "Q"):
            return RuleVerdict(
                factor=0.15,
                note=f"引入 {mutant} 会新增脱酰胺位点，与耐碱性改造目标相冲突",
                flags=["新增脱酰胺风险位点"],
            )

        return RuleVerdict(factor=0.5, note=f"{wild_type} -> {mutant}：对亲和力与耐碱性均无明显影响")

    # ---------------- 策略说明 ---------------- #
    def design_hints(self, sequence: str) -> list[str]:
        units, identity, conserved = self._analyze(sequence)
        counts = composition_counts(sequence)
        hints: list[str] = []

        if units:
            hints.append(
                f"检测到 {len(units)} 个串联重复单元，结构域长度约 {units[0].length} aa，"
                f"单元间平均一致性 {identity:.0%}（IgG 结合结构域的典型特征）。"
            )
            hints.append(
                f"已保护 {len(conserved)} 个跨重复保守位点（框架与 Fc 结合界面），"
                "这些位点改造需以实测 KD 为准。"
            )
        else:
            hints.append(
                "未检测到串联重复结构域，可能缺少 Ig 结合结构域或序列被截断；"
                "保守位点保护未生效，请人工核对。"
            )

        hints.append(
            f"残基组成：Asn {counts['N']} 个、Gln {counts['Q']} 个。"
            "耐碱性提升的核心是把 Asn（尤其 Asn-Gly/Asn-Ser）替换为 Gln/Asp/Thr/Ser/Ala。"
        )
        hints.append(
            "典型的碱稳定改造顺序：先消除全部 Asn-Gly 基序（收益最高），"
            "再处理暴露的 Asn-Ser/Asn-Thr，最后评估 Gln；每轮改造后须实测 0.1 M NaOH 处理后的结合容量。"
        )
        hints.append(
            "亲和力保持：Fc 结合界面位于螺旋 I/II 的疏水面，"
            "不要在跨重复保守的芳香族/疏水残基上做替换。"
        )
        return hints

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["alkali_preferred"] = sorted(self.alkali_preferred)
        payload["detection"] = "串联重复检测 + 跨重复保守位点分析"
        return payload
