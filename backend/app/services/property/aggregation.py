"""聚集倾向预测。

算法依据
--------
聚集（aggregation）与淀粉样纤维形成的公认驱动因素：

1. **疏水斑块**：短窗口内的高平均疏水性是聚集的核心驱动（Chiti & Dobson 2006）。
2. **β-折叠倾向**：TANGO / Zyggregator 类方法的核心成分——只有同时具备"疏水 +
   高 β 倾向"的片段才是真正的聚集热点，单纯疏水不构成瓶颈。
3. **芳香族残基**：π-π 堆积稳定纤维核心。
4. **低复杂度 / 重复区**：Q/N 富集区与淀粉样聚集高度相关。
5. **表面暴露度**：若结构可用，暴露的聚集热点比埋藏的更具风险。

实现为**逐残基滑窗打分**，输出完整序列的聚集倾向轨道（供前端位点轨道图渲染），
再由轨道汇总为 0-100 的总分。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...core.logging import get_logger
from ..sequence.feature_utils import (
    KYTE_DOOLITTLE,
    SHEET_PROPENSITY,
    low_complexity_score,
)
from .metric import Evidence, Metric, aggregate_weighted, linear_score

logger = get_logger(__name__)

#: 判定为聚集热点的逐残基分数阈值
HOTSPOT_THRESHOLD = 62.0
#: 聚集热点区段合并时允许的最大间隔
SEGMENT_GAP = 2


@dataclass
class AggregationSegment:
    """一段聚集倾向区段。"""

    start: int
    end: int
    peak_score: float
    mean_score: float
    sequence: str

    @property
    def length(self) -> int:
        return self.end - self.start


def aggregation_profile(sequence: str, window: int = 7) -> list[float]:
    """逐残基聚集倾向（0-100）。

    每个位置取以它为中心的滑窗，计算"疏水性 × β-折叠倾向"的乘积分数。
    两个因子都高时才会得到高分，避免把纯疏水但柔性（如富含 Gly/Pro）的区域
    误判为聚集热点。
    """
    length = len(sequence)
    if length == 0:
        return []

    half = window // 2
    profile: list[float] = []

    for index in range(length):
        start = max(0, index - half)
        end = min(length, start + window)
        start = max(0, end - window)
        segment = sequence[start:end]

        mean_hydro = sum(KYTE_DOOLITTLE.get(char, 0.0) for char in segment) / len(segment)
        mean_beta = sum(SHEET_PROPENSITY.get(char, 0.9) for char in segment) / len(segment)

        # Kyte-Doolittle 有效范围约 -4.5 ~ 4.5；β 倾向约 0.37 ~ 1.70
        hydro_norm = max(0.0, min(1.0, (mean_hydro + 0.5) / 4.0))
        beta_norm = max(0.0, min(1.0, (mean_beta - 0.70) / 0.80))

        profile.append(round(100.0 * hydro_norm * beta_norm, 2))

    return profile


def find_aggregation_segments(
    profile: list[float], sequence: str, threshold: float = HOTSPOT_THRESHOLD
) -> list[AggregationSegment]:
    """把逐残基轨道合并为热点区段。"""
    segments: list[AggregationSegment] = []
    start: int | None = None
    gap = 0

    for index, score in enumerate(profile):
        if score >= threshold:
            if start is None:
                start = index
            gap = 0
        elif start is not None:
            gap += 1
            if gap > SEGMENT_GAP:
                segments.append(_build_segment(profile, sequence, start, index - gap + 1))
                start = None
                gap = 0

    if start is not None:
        segments.append(_build_segment(profile, sequence, start, len(profile)))

    return sorted(segments, key=lambda item: -item.peak_score)


def _build_segment(profile: list[float], sequence: str, start: int, end: int) -> AggregationSegment:
    values = profile[start:end]
    return AggregationSegment(
        start=start,
        end=end,
        peak_score=round(max(values), 2) if values else 0.0,
        mean_score=round(sum(values) / len(values), 2) if values else 0.0,
        sequence=sequence[start:end],
    )


def assess_aggregation(
    sequence: str,
    biophys_values: dict[str, Any],
    structure_stats: dict[str, Any] | None = None,
) -> Metric:
    """聚集风险评分：分数越高表示聚集风险越低（越安全）。"""
    length = max(1, len(sequence))
    counts = biophys_values.get("counts", {})
    groups = biophys_values.get("group_ratios", {})

    profile = aggregation_profile(sequence)
    segments = find_aggregation_segments(profile, sequence)
    peak = max(profile) if profile else 0.0
    hotspot_count = sum(1 for value in profile if value >= HOTSPOT_THRESHOLD)

    aromatic = float(groups.get("aromatic", 0.0))
    low_complexity = low_complexity_score(sequence)

    # 暴露度修正：结构可用时，把"暴露的疏水核心"作为额外风险
    exposed_hydrophobic: float | None = None
    if structure_stats:
        exposure = structure_stats.get("hydrophobic_exposure", {})
        if isinstance(exposure, dict) and exposure.get("exposed_ratio") is not None:
            exposed_hydrophobic = float(exposure["exposed_ratio"])
            # 若热点区段落在暴露位置，风险上调（这里做整体近似）
            profile_exposed = [
                profile[index]
                for index in (exposure.get("exposed_positions") or [])
                if 0 <= index < len(profile)
            ]
            if profile_exposed:
                peak = max(peak, max(profile_exposed))

    parts = [
        (
            0.30,
            linear_score(peak, worst=85.0, best=30.0),
            Evidence(
                label="最高聚集倾向（逐残基峰值）",
                value=round(peak, 2),
                rationale="峰值体现最危险的单一聚集热点；>62 判定为热点区段。",
            ),
        ),
        (
            0.22,
            linear_score(float(hotspot_count), worst=max(8.0, length * 0.06), best=0.0),
            Evidence(
                label="聚集热点残基数",
                value=hotspot_count,
                rationale="热点残基数量越多，可供分子间 β-折叠配对的机会越多。",
            ),
        ),
        (
            0.16,
            linear_score(float(len(segments)), worst=5.0, best=0.0),
            Evidence(
                label="聚集热点区段数",
                value=len(segments),
                rationale="多个独立热点区段比单一热点更难通过单点突变消除。",
            ),
        ),
        (
            0.12,
            linear_score(aromatic, worst=0.12, best=0.04),
            Evidence(
                label="芳香族残基占比",
                value=round(aromatic, 4),
                rationale="Phe/Tyr/Trp 通过 π-π 堆积稳定淀粉样纤维核心；占比高时聚集风险上升。",
            ),
        ),
        (
            0.10,
            linear_score(low_complexity, worst=0.35, best=0.0),
            Evidence(
                label="低复杂度区域占比",
                value=round(low_complexity, 4),
                rationale="单一残基主导的低复杂度区（尤其 Q/N 富集）与淀粉样聚集高度相关。",
            ),
        ),
        (
            0.10,
            linear_score(exposed_hydrophobic, worst=0.35, best=0.08)
            if exposed_hydrophobic is not None
            else None,
            Evidence(
                label="疏水核心暴露比例",
                value=round(exposed_hydrophobic, 4) if exposed_hydrophobic is not None else "不可用",
                rationale="折叠松散导致疏水核心暴露，是聚集的直接结构前提。",
            ),
        ),
    ]

    score, evidence = aggregate_weighted(parts)
    loci = [
        {
            "position": segment.start,
            "end": segment.end,
            "length": segment.length,
            "peak": segment.peak_score,
            "mean": segment.mean_score,
            "type": "aggregation_hotspot",
            "severity": round(min(1.0, segment.peak_score / 100.0), 2),
            "sequence": segment.sequence,
        }
        for segment in segments
    ]

    return Metric(
        key="aggregation",
        label="聚集风险",
        score=round(score, 1),
        algorithm="逐残基滑窗（疏水性 × Chou-Fasman β 倾向）+ 热点区段聚合 + 芳香族/低复杂度修正",
        rationale=(
            "分数越高表示聚集风险越低。若存在高分热点区段，"
            "改造优先级为：把热点内的疏水/β 倾向残基替换为带电或脯氨酸残基（如 V→E、I→K、F→P）。"
        ),
        confidence=0.6,
        evidence=evidence,
        locus=loci[:200],
        meta={
            "peak_score": round(peak, 2),
            "hotspot_count": hotspot_count,
            "profile": profile,  # 完整轨道，供前端位点轨道图渲染
        },
    )


def aggregation_track(sequence: str) -> list[dict[str, Any]]:
    """返回紧凑的轨道数据（前端直接用）。"""
    profile = aggregation_profile(sequence)
    return [
        {"position": index, "residue": sequence[index], "score": value}
        for index, value in enumerate(profile)
    ]
