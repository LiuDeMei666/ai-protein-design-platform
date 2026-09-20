"""重组蛋白酶（枯草杆菌蛋白酶家族）规则包。

领域知识
--------
1. **催化三联体 Asp-His-Ser** 与 **氧负离子洞 Asn** 绝对不可动。
2. 枯草杆菌蛋白酶（subtilisin）的活性位点基序高度保守，可用序列模体识别，
   **不需要依赖绝对残基编号**（前体与成熟酶编号相差 107 位，跨来源直接套用编号
   是常见错误）：
   * 催化 Ser：模体 ``GTSM``（BPN' 第 221 位 Ser 所在）
   * 催化 Asp：模体 ``DGN``（第 32 位）
   * 催化 His：模体 ``HGT``（第 64 位）
   * 氧负离子洞 Asn：模体 ``NNS``（第 155 位）
3. **热稳定性改造**：嗜热同源物中富集 E/K/R/Y/F/W/P，而 G/N/Q/C/M 偏少——
   这是本项目配置里 thermostability_bias 的来源，用于给替换打方向性偏好。
4. **pH 适应范围拓宽**：通过表面电荷重排（D/E ↔ K/R）改变活性中心的静电环境，
   是工业蛋白酶改造中最常用的策略之一。
5. **底物特异性改造**：主要改动 S1-S4 底物结合口袋（序列上难以精确定位，
   需要结构信息；本模块在检测到活性位点后按相对编号映射配置中的口袋位点）。
"""

from __future__ import annotations

import re
from typing import Any

from ....core.config import load_platform_config
from ....core.logging import get_logger
from ...sequence.residue_properties import (
    THERMOSTABLE_DISCOURAGED,
    THERMOSTABLE_PREFERRED,
)
from .base import RuleContext, RulePack, RuleVerdict

logger = get_logger(__name__)

#: 枯草杆菌蛋白酶家族活性位点模体 -> (残基在模体中的偏移, 角色, 说明)
#:
#: 模体经对 UniProt P00782（枯草杆菌蛋白酶 BPN' 前体，382 aa）实测校准，
#: 每个模体在该序列中**唯一出现**，且位置与 BPN' 成熟酶编号推算完全吻合：
#: DSG@138(Asp32)、HGTH@170(His64)、GTSM@325+2(Ser221)、GNEGT@261(Asn155)
#:
#: 注意：早期版本误用 ``DGN`` 与 ``NNS`` 作为模体——前者在序列中根本不存在，
#: 后者出现两次导致过度保护。模体必须实测校准，不能凭印象填写。
ACTIVE_SITE_MOTIFS: dict[str, tuple[int, str, str]] = {
    "DSG": (0, "asp", "催化天冬氨酸 Asp（Asp32，BPN' 编号）"),
    "HGTH": (0, "his", "催化组氨酸 His（His64，BPN' 编号）"),
    "GTSM": (2, "ser", "催化丝氨酸 Ser（Ser221，BPN' 编号）"),
    "GNEGT": (0, "asn", "氧负离子洞天冬酰胺（催化 Ser 上游 67 位，BPN' 家族保守）"),
}

#: 催化 Ser 与其它活性位点之间的**相对间距**（BPN' 编号差）。
#: 相对间距与序列编号体系无关，因此可作为"是否真的是枯草杆菌蛋白酶活性位点"的
#: 自校验依据——避免在随机序列上误判出假的活性位点。
EXPECTED_SPACING: dict[str, int] = {
    "ser": 0,
    "his": 221 - 64,   # 157
    "asp": 221 - 32,   # 189
    "asn": 221 - 155,  # 66
}
SPACING_TOLERANCE = 2

#: BPN' 成熟酶第 1 位在 382 aa 前体中的 0-based 序号
BPN_MATURE_OFFSET = 107

#: 活性位点周围的"敏感半径"（残基），半径内改动的活性风险显著上升
SENSITIVE_RADIUS = 8


def detect_active_site(sequence: str) -> tuple[dict[int, str], int | None]:
    """通过序列模体识别活性位点，并用相对间距自校验。

    Returns:
        ``({0-based 位点: 说明}, 催化 Ser 的 0-based 位点或 None)``
    """
    hits: dict[str, int] = {}
    for motif, (offset, role, _label) in ACTIVE_SITE_MOTIFS.items():
        matches = [match.start() + offset for match in re.finditer(f"(?={motif})", sequence)]
        matches = [position for position in matches if position < len(sequence)]
        if not matches:
            continue
        # 同一模体多次出现时，选取与已定位角色相对间距最吻合的一个
        if role == "ser" or not hits:
            hits[role] = matches[0]
            continue
        reference = hits.get("ser")
        if reference is None:
            hits[role] = matches[0]
            continue
        expected = EXPECTED_SPACING.get(role)
        if expected is None:
            hits[role] = matches[0]
            continue
        # Asp / Asn 在 Ser 之前，间距为 reference - position
        best = min(matches, key=lambda position: abs((reference - position) - expected))
        hits[role] = best

    catalytic_ser = hits.get("ser")
    protected: dict[int, str] = {}

    for role, position in hits.items():
        _motif, (_offset, _role, label) = next(
            (item for item in ACTIVE_SITE_MOTIFS.items() if item[1][1] == role),
            ("?", (0, role, role)),
        )
        protected[position] = label

    # ---------- 相对间距自校验 ----------
    if catalytic_ser is not None:
        mismatched: list[str] = []
        for role in ("his", "asp", "asn"):
            position = hits.get(role)
            expected = EXPECTED_SPACING[role]
            if position is None:
                mismatched.append(f"{role}(未检出)")
                continue
            actual = catalytic_ser - position
            if abs(actual - expected) > SPACING_TOLERANCE:
                mismatched.append(f"{role}(间距 {actual} vs 期望 {expected})")
        if mismatched:
            logger.warning(
                "活性位点相对间距校验未通过：%s。"
                "该序列可能不是标准枯草杆菌蛋白酶，催化残基保护需人工复核。",
                "、".join(mismatched),
            )

    return protected, catalytic_ser


def verify_active_site(sequence: str) -> dict[str, Any]:
    """返回活性位点识别的可核对明细（用于报告与文档）。"""
    protected, catalytic_ser = detect_active_site(sequence)
    spacing: dict[str, Any] = {}
    if catalytic_ser is not None:
        for role, expected in EXPECTED_SPACING.items():
            position = next(
                (key for key, value in protected.items() if _role_of(value) == role), None
            )
            if position is None:
                spacing[role] = {"expected": expected, "actual": None, "ok": False}
                continue
            actual = catalytic_ser - position
            spacing[role] = {
                "expected": expected,
                "actual": actual,
                "ok": abs(actual - expected) <= SPACING_TOLERANCE,
            }
    return {
        "catalytic_ser_0based": catalytic_ser,
        "catalytic_ser_1based": catalytic_ser + 1 if catalytic_ser is not None else None,
        "protected": {position + 1: reason for position, reason in protected.items()},
        "expected_spacing": EXPECTED_SPACING,
        "spacing_check": spacing,
        "all_checks_passed": bool(spacing) and all(item["ok"] for item in spacing.values()),
    }


def _role_of(reason: str) -> str:
    if "丝氨酸" in reason:
        return "ser"
    if "组氨酸" in reason:
        return "his"
    if "天冬氨酸" in reason:
        return "asp"
    if "天冬酰胺" in reason:
        return "asn"
    return "unknown"


class ProteaseRulePack(RulePack):
    """重组蛋白酶专用规则。"""

    category = "protease"
    label = "重组蛋白酶"
    description = "催化效率(kcat/Km)、pH/温度适应范围与底物特异性改造"

    def __init__(self) -> None:
        config = load_platform_config().get("rulepacks", {}).get("protease", {})
        bias = config.get("thermostability_bias", {})
        self.preferred_pool = set(bias.get("preferred", list(THERMOSTABLE_PREFERRED)))
        self.discouraged_pool = set(bias.get("discouraged", list(THERMOSTABLE_DISCOURAGED)))
        # 前导肽区保护：默认开启。若序列本身就是成熟酶（无 proregion），
        # 检测不到催化 Ser 时该逻辑自然不生效；若检测到但序列确为成熟酶，
        # 可通过配置关闭。
        self._protect_proregion = bool(config.get("protect_proregion", True))
        self._cache: dict[str, tuple[dict[int, str], int | None]] = {}

    # ---------------- 活性位点 ---------------- #
    def _active_site(self, sequence: str) -> tuple[dict[int, str], int | None]:
        if sequence not in self._cache:
            self._cache[sequence] = detect_active_site(sequence)
            if not self._cache[sequence][0]:
                logger.warning(
                    "未在序列中识别到枯草杆菌蛋白酶活性位点模体，"
                    "催化残基保护将失效，请人工核对序列完整性"
                )
        return self._cache[sequence]

    def protected_positions(self, sequence: str) -> dict[int, str]:
        """保护前导肽区、催化三联体、氧负离子洞与底物口袋。"""
        protected, catalytic_ser = self._active_site(sequence)

        # 若检出催化 Ser，则按其位置推导 BPN' 编号偏移，映射配置中的口袋位点
        if catalytic_ser is not None:
            offset = catalytic_ser - 220  # BPN' 编号 221 -> 0-based 220
            config = load_platform_config().get("rulepacks", {}).get("protease", {})
            for label, key in (("底物结合口袋", "substrate_pocket"),):
                for entry in config.get(key, []) or []:
                    position = _parse_bpn_position(entry) + offset
                    if position is None or not (0 <= position < len(sequence)):
                        continue
                    if position in protected:
                        continue
                    protected[position] = (
                        f"{label}残基（{entry}，BPN' 编号），直接决定底物识别与 kcat/Km，"
                        "改造需有结构依据"
                    )

            # 前导肽区（信号肽 + propeptide）在成熟过程中被自切去除，
            # 在这一区段推荐突变对成熟酶毫无意义——必须显式保护。
            # 依据：催化 Ser 位于成熟酶第 221 位，故成熟酶第 1 位的前一位
            # 即为 proregion 的末端；precursor 0-based 索引 = catalytic_ser - 220 - 1。
            if self._protect_proregion:
                mature_start = catalytic_ser - 220
                for position in range(0, max(0, min(mature_start, len(sequence)))):
                    protected.setdefault(
                        position,
                        "信号肽/前导肽（proregion）区段，成熟过程中被自切去除，"
                        "在该区段推荐突变对成熟酶无意义",
                    )
        return protected

    # ---------------- 活性偏好 ---------------- #
    def preferred_residues(self, sequence: str, position: int) -> set[str]:
        """表面/非活性位点优先推荐嗜热型残基。"""
        protected, catalytic_ser = self._active_site(sequence)
        if catalytic_ser is not None and abs(position - catalytic_ser) <= SENSITIVE_RADIUS:
            return set()
        return set(self.preferred_pool)

    def discouraged_residues(self, sequence: str, position: int) -> set[str]:
        return set(self.discouraged_pool)

    def activity_factor(self, rule_context: RuleContext) -> RuleVerdict:
        sequence = rule_context.sequence
        position = rule_context.position
        mutant = rule_context.mutant
        protected, catalytic_ser = self._active_site(sequence)

        if position in protected:
            return RuleVerdict(
                factor=0.0,
                note=f"该位点受保护：{protected[position]}",
                flags=["触及活性必需位点"],
            )

        distance: int | None = None
        if catalytic_ser is not None:
            distance = abs(position - catalytic_ser)

        if distance is not None and distance <= 4:
            return RuleVerdict(
                factor=0.05,
                note=f"距催化丝氨酸仅 {distance} 个残基，改动极可能破坏催化几何或氧负离子洞",
                flags=["邻近催化中心"],
                evidence={"distance_to_catalytic_ser": distance},
            )
        if distance is not None and distance <= SENSITIVE_RADIUS:
            if mutant in THERMOSTABLE_PREFERRED:
                return RuleVerdict(
                    factor=0.55,
                    note=(
                        f"距催化丝氨酸 {distance} 个残基（活性位点敏感区），"
                        f"替换为 {mutant} 属于嗜热型残基，可能在提升稳定性的同时保持活性，"
                        "但须实验验证 kcat/Km"
                    ),
                    evidence={"distance_to_catalytic_ser": distance, "bias": "thermostable_preferred"},
                )
            return RuleVerdict(
                factor=0.28,
                note=f"距催化丝氨酸 {distance} 个残基，位于活性位点敏感区，活性风险较高",
                flags=["邻近催化中心"],
                evidence={"distance_to_catalytic_ser": distance},
            )

        # 远离活性中心：按嗜热改造偏好给分
        if mutant in self.preferred_pool:
            return RuleVerdict(
                factor=0.85,
                note=(
                    f"远离活性中心（距催化丝氨酸 {distance if distance is not None else '未知'} 个残基），"
                    f"替换为 {mutant} 属于嗜热同源物富集的残基，"
                    "有利于拓宽温度适应范围（建议同时核对表面可及性）"
                ),
                evidence={"bias": "thermostable_preferred"},
            )
        if mutant in self.discouraged_pool:
            return RuleVerdict(
                factor=0.32,
                note=(
                    f"替换为 {mutant}，该残基在嗜热同源物中偏少，"
                    "可能降低热稳定性（G/N/Q 还带来脱酰胺风险，C/M 带来氧化风险）"
                ),
                evidence={"bias": "thermostable_discouraged"},
            )
        return RuleVerdict(factor=0.55, note=f"替换为 {mutant}，不属于明显的稳定化或去稳定化方向")

    # ---------------- 策略说明 ---------------- #
    def design_hints(self, sequence: str) -> list[str]:
        verification = verify_active_site(sequence)
        hints: list[str] = []

        if verification["catalytic_ser_1based"]:
            details = "；".join(
                f"第 {position} 位 {reason.split('（')[0]}"
                for position, reason in verification["protected"].items()
            )
            hints.append(f"已按序列模体识别活性位点：{details}")
            checks = verification["spacing_check"]
            passed = [role for role, item in checks.items() if item["ok"]]
            failed = [role for role, item in checks.items() if not item["ok"]]
            hints.append(
                f"相对间距自校验：{len(passed)}/{len(checks)} 项通过"
                + (f"；未通过项 {', '.join(failed)}（该序列可能非标准枯草杆菌蛋白酶）" if failed else "")
            )
        else:
            hints.append(
                "未识别到枯草杆菌蛋白酶活性位点模体（DSG/HGTH/GTSM/GNEGT），"
                "可能是非典型序列或片段，活性位点保护未生效，请人工核对。"
            )

        hints.extend(
            [
                "提升 kcat/Km：优先改造底物结合口袋（S1-S4）与表面环区，"
                "严禁改动催化三联体与氧负离子洞；建议以「表面 + 偏离活性中心」为筛选前提。",
                "拓宽 pH 适应范围：在远离活性中心处做表面电荷重排（D/E ↔ K/R），"
                "改变活性中心局部静电环境；建议成对改换以维持整体电荷平衡。",
                "拓宽温度适应范围：在表面位点引入 E/K/R/Y/F/W/P（嗜热同源物富集残基），"
                "并优先选择埋藏良好的位点做填充型替换。",
                "改造底物特异性：S1 口袋底部的残基决定 P1 侧链偏好，"
                "建议先预测结构确认口袋位置，再针对口袋内壁做定点改造。",
            ]
        )
        return hints

    def describe(self, sequence: str | None = None) -> dict[str, Any]:
        payload = super().describe()
        payload["active_site_motifs"] = {
            motif: label for motif, (_, _, label) in ACTIVE_SITE_MOTIFS.items()
        }
        payload["mature_offset"] = BPN_MATURE_OFFSET
        payload["spacing_self_check"] = EXPECTED_SPACING
        if sequence:
            payload["active_site_verification"] = verify_active_site(sequence)
        return payload


def _parse_bpn_position(entry: str) -> int | None:
    """从 ``"S221"`` 这类 BPN' 编号中取出 0-based 序号。"""
    match = re.search(r"(\d+)", str(entry))
    if match is None:
        return None
    return int(match.group(1)) - 1
