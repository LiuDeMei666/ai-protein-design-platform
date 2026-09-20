"""突变位点枚举与扫描范围规划。

职责
----
1. 决定"哪些位点可以突变"：排除末端、排除规则包保护的位点、遵守用户给定的
   区段限制与 `max_positions` 上限。
2. **绝不静默丢弃**：每一个被排除的位点都会记录排除原因，前端可以展示
   "为什么这个位点没被推荐"，这对实验人员判断结果完整性至关重要。
3. 当位点数超过上限时，用 ESM-2 的**未掩码对数概率**做一次极廉价的预筛
   （单次前向），优先保留模型认为"可变动空间大"的位点，并显式记录降采样行为。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...core.config import load_platform_config
from ...core.logging import describe_sequence, get_logger
from ..sequence.feature_utils import composition_counts
from ..sequence.validator import validate_sequence

logger = get_logger(__name__)

#: 20 种标准氨基酸（与 ESM-2 字母表一致）
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class ExclusionRecord:
    """一个被排除的位点及其原因。"""

    position: int
    residue: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"position": self.position, "residue": self.residue, "reason": self.reason}


@dataclass
class ScanPlan:
    """扫描计划。"""

    sequence: str
    positions: list[int]
    exclusions: list[ExclusionRecord] = field(default_factory=list)
    protected: dict[int, str] = field(default_factory=dict)
    downsampled: bool = False
    downsample_note: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def candidate_count(self) -> int:
        """候选突变总数（位点数 × 19）。"""
        return len(self.positions) * 19

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_count": len(self.positions),
            "candidate_count": self.candidate_count,
            "positions": self.positions,
            "exclusions": [item.to_dict() for item in self.exclusions],
            "protected": {str(key): value for key, value in self.protected.items()},
            "downsampled": self.downsampled,
            "downsample_note": self.downsample_note,
            "notes": self.notes,
        }


def default_positions(
    sequence: str,
    *,
    exclude_terminal: int = 1,
    region: tuple[int, int] | None = None,
    max_positions: int = 1200,
) -> tuple[list[int], list[ExclusionRecord]]:
    """基础可用位点集合（未考虑规则包保护）。"""
    length = len(sequence)
    exclusions: list[ExclusionRecord] = []

    start, end = (0, length) if region is None else (max(0, region[0]), min(length, region[1]))

    for index in range(length):
        if index < start or index >= end:
            exclusions.append(
                ExclusionRecord(index, sequence[index], "超出用户指定的突变区段")
            )
        elif index < exclude_terminal or index >= length - exclude_terminal:
            exclusions.append(
                ExclusionRecord(index, sequence[index], f"位于序列末端 {exclude_terminal} 位内，突变会干扰翻译起始/终止")
            )

    positions = [
        index
        for index in range(start, end)
        if exclude_terminal <= index < length - exclude_terminal
    ]
    return positions, exclusions


def downsample_by_model(
    sequence: str,
    positions: list[int],
    max_positions: int,
    log_probs: np.ndarray,
) -> tuple[list[int], str]:
    """用未掩码对数概率预筛位点。

    保留依据：野生型残基对数概率越低（模型越"意外"），说明该位点受上下文约束
    越弱、越可能容忍替换。这是**廉价的预筛启发式**（单次前向），
    最终打分仍由掩码边缘完成。
    """
    scored: list[tuple[float, int]] = []
    for position in positions:
        residue = sequence[position]
        if residue not in AA_ORDER:
            continue
        value = float(log_probs[position, AA_ORDER.index(residue)])
        if np.isnan(value):
            value = -3.0
        scored.append((value, position))

    scored.sort(key=lambda item: item[0])
    kept = sorted(position for _, position in scored[:max_positions])
    note = (
        f"候选位点 {len(scored)} 个超过上限 {max_positions}，"
        f"已用 ESM-2 单次前向的野生型对数概率做预筛，"
        f"保留模型约束最弱（最可能容忍突变）的 {len(kept)} 个位点。"
        f"**其余 {len(scored) - len(kept)} 个位点未参与扫描**，如需全位点覆盖请分区段多次扫描。"
    )
    return kept, note


def build_scan_plan(
    sequence: str,
    *,
    region: tuple[int, int] | None = None,
    max_positions: int | None = None,
    protected: dict[int, str] | None = None,
    strict: bool = True,
    exclude_terminal: int | None = None,
) -> ScanPlan:
    """构建扫描计划。

    Args:
        sequence: 目标序列。
        region: 限定突变区段 ``(start, end)``（0-based 半开）。
        max_positions: 位点上限，默认取配置 ``scan.max_positions``。
        protected: 规则包给出的受保护位点 ``{position: 原因}``。
        strict: 是否要求序列全部为标准氨基酸（突变设计必须为 ``True``）。
        exclude_terminal: 两端排除的残基数，默认取配置。
    """
    config = load_platform_config().get("scan", {})
    limit = max_positions if max_positions is not None else int(config.get("max_positions", 1200))
    terminal = (
        exclude_terminal
        if exclude_terminal is not None
        else int(config.get("exclude_terminal", 1))
    )

    check = validate_sequence(sequence, strict=strict)
    if not check.ok:
        from ...core.errors import SequenceError

        raise SequenceError("; ".join(check.errors), detail={"warnings": check.warnings})

    cleaned = check.sequence
    protected = protected or {}

    positions, exclusions = default_positions(
        cleaned, exclude_terminal=terminal, region=region, max_positions=limit
    )

    # 规则包保护位点
    allowed: list[int] = []
    for position in positions:
        if position in protected:
            exclusions.append(
                ExclusionRecord(position, cleaned[position], protected[position])
            )
        else:
            allowed.append(position)

    plan = ScanPlan(
        sequence=cleaned,
        positions=allowed,
        exclusions=exclusions,
        protected=dict(protected),
        notes=list(check.warnings),
    )

    if len(allowed) > limit:
        # 需要预筛：调用 ESM-2 单次前向（失败时按等间隔抽样，仍然显式记录）
        try:
            from ..embedding.esm2 import get_service

            log_probs = get_service().wild_type_logprobs(cleaned)
            kept, note = downsample_by_model(cleaned, allowed, limit, log_probs)
            plan.positions = kept
            plan.downsampled = True
            plan.downsample_note = note
            plan.notes.append(note)
        except Exception as exc:
            step = max(1, len(allowed) // limit)
            kept = allowed[::step][:limit]
            note = (
                f"候选位点 {len(allowed)} 个超过上限 {limit}，"
                f"且 ESM-2 预筛不可用（{exc}），已按等间隔抽样保留 {len(kept)} 个位点。"
            )
            plan.positions = kept
            plan.downsampled = True
            plan.downsample_note = note
            plan.notes.append(note)
            logger.warning("ESM-2 预筛失败，退化为等间隔抽样: %s", exc)

    logger.info(
        "扫描计划 %s 位点=%d 候选=%d 排除=%d 降采样=%s",
        describe_sequence(cleaned),
        len(plan.positions),
        plan.candidate_count,
        len(plan.exclusions),
        plan.downsampled,
    )
    return plan


def build_mutant_sequence(sequence: str, position: int, mutant: str) -> str:
    """构造单点突变序列。"""
    return sequence[:position] + mutant + sequence[position + 1 :]


def build_combinatorial_sequence(sequence: str, mutations: dict[int, str]) -> str:
    """构造多点组合突变序列。"""
    chars = list(sequence)
    for position, mutant in mutations.items():
        if 0 <= position < len(chars):
            chars[position] = mutant
    return "".join(chars)


def mutation_label(sequence: str, position: int, mutant: str) -> str:
    """生成标准突变标签，如 ``A123V``（位置从 1 开始计数）。"""
    wild_type = sequence[position] if 0 <= position < len(sequence) else "X"
    return f"{wild_type}{position + 1}{mutant}"


@dataclass
class PositionContext:
    """单个位点的结构先验上下文。"""

    position: int
    residue: str
    plddt: float | None = None
    relative_sasa: float | None = None
    secondary_structure: str | None = None

    @property
    def is_buried(self) -> bool:
        threshold = float(
            load_platform_config().get("design", {}).get("risk", {}).get("buried_sasa_threshold", 0.15)
        )
        return self.relative_sasa is not None and self.relative_sasa < threshold

    @property
    def is_helix(self) -> bool:
        return self.secondary_structure in ("H", "G", "I")

    @property
    def is_sheet(self) -> bool:
        return self.secondary_structure == "E"

    def describe(self) -> str:
        parts: list[str] = []
        if self.plddt is not None:
            parts.append(f"pLDDT={self.plddt:.0f}")
        if self.relative_sasa is not None:
            parts.append(f"相对SASA={self.relative_sasa:.2f}")
        if self.secondary_structure:
            label = {
                "H": "α-螺旋", "G": "3-10螺旋", "I": "π-螺旋",
                "E": "β-折叠", "-": "无规卷曲",
            }.get(self.secondary_structure, self.secondary_structure)
            parts.append(label)
        return "，".join(parts) if parts else "无结构信息"


def build_position_contexts(
    sequence: str, structure_stats: dict[str, Any] | None
) -> dict[int, PositionContext]:
    """由结构统计构建逐位点上下文；无结构时返回空字典（调用方需容忍缺省）。"""
    contexts: dict[int, PositionContext] = {}
    if not structure_stats or structure_stats.get("stub"):
        return contexts

    plddt_values: list[float] = structure_stats.get("plddt") or []
    sasa_values: list[float] = structure_stats.get("relative_sasa") or []
    secondary = str(structure_stats.get("secondary_structure") or "")

    for index, residue in enumerate(sequence):
        contexts[index] = PositionContext(
            position=index,
            residue=residue,
            plddt=float(plddt_values[index]) if index < len(plddt_values) else None,
            relative_sasa=float(sasa_values[index]) if index < len(sasa_values) else None,
            secondary_structure=secondary[index] if index < len(secondary) else None,
        )
    return contexts


def residue_composition_note(sequence: str) -> dict[str, int]:
    """组成摘要，便于前端在参数面板展示。"""
    return composition_counts(sequence)
