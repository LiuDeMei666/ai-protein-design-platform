"""多点组合突变搜索（束搜索 + 两阶段精评）。

为什么需要两阶段
----------------
组合突变的准确打分必须考虑**上位效应**（epistasis）：某个突变在野生型背景下有利，
在另一个突变已存在的背景下可能无益甚至有害。准确的算法是"对每个突变，把它在
组合背景下重新掩码打分"，代价是 ``组合数 × 位点数`` 次前向。

朴素做法（对全部组合都做精确评估）在 15 个候选取 3 个时就是
``C(15,3)=455`` 个组合 × 3 次前向 ≈ 1365 次前向，耗时以分钟计。因此：

1. **阶段一（廉价）**：用单点分数相加 + 上位效应惩罚，估计全部组合；
2. **阶段二（精确）**：只对估计排序前 ``refine_top_k`` 的组合做上下文
   依赖的真实重打分。

这样既保证最终给出的组合是经过严格评估的，又把代价压到可接受范围。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

from ...core.config import load_platform_config
from ...core.logging import get_logger
from .scanner import build_combinatorial_sequence

logger = get_logger(__name__)


@dataclass
class SingleMutation:
    """参与组合的单点突变。"""

    position: int
    mutant: str
    score: float
    label: str


@dataclass
class CombinationCandidate:
    """一个组合候选。"""

    mutations: list[SingleMutation]
    estimated_score: float
    refined_score: float | None = None
    epitasis_penalty: float = 0.0
    refined: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return [item.label for item in self.mutations]

    @property
    def positions(self) -> list[int]:
        return [item.position for item in self.mutations]

    @property
    def final_score(self) -> float:
        return self.refined_score if self.refined_score is not None else self.estimated_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutations": self.labels,
            "positions": self.positions,
            "estimated_score": round(self.estimated_score, 2),
            "refined_score": round(self.refined_score, 2) if self.refined_score is not None else None,
            "final_score": round(self.final_score, 2),
            "epitasis_penalty": round(self.epitasis_penalty, 2),
            "refined": self.refined,
            "details": self.details,
        }


def _epitasis_penalty(
    positions: list[int], min_distance: int, penalty_coefficient: float
) -> float:
    """位点过近时的上位效应惩罚。

    结构上相邻的位点会互相扰动局部环境，单点效应不可简单叠加，
    因此对间距小于 ``min_distance`` 的位点对施加惩罚。
    """
    penalty = 0.0
    for first, second in itertools.combinations(sorted(positions), 2):
        gap = second - first
        if gap < min_distance:
            # 间距越小惩罚越重：gap=1 时满额，达到 min_distance 时归零
            severity = (min_distance - gap) / max(1, min_distance - 1)
            penalty += penalty_coefficient * severity
    return penalty


def search_combinations(
    singles: list[SingleMutation],
    *,
    max_sites: int | None = None,
    beam_width: int | None = None,
    pool_size: int | None = None,
    refine_top_k: int = 12,
    refine_fn: Callable[[dict[int, str], "CombinationCandidate"], float] | None = None,
) -> list[CombinationCandidate]:
    """束搜索 + 两阶段精评。

    Args:
        singles: 已排序（分数降序）的单点突变列表。
        max_sites: 单个组合最多叠加的位点数。
        beam_width: 束宽。
        pool_size: 参与组合的单点数量上限。
        refine_top_k: 用 ``refine_fn`` 精确重评分的组合数。
        refine_fn: 精确评分函数，签名 ``(mutations: {position: mutant}, candidate) -> score``。
    """
    config = load_platform_config().get("combination", {})
    max_sites = max_sites or int(config.get("max_sites", 3))
    beam_width = beam_width or int(config.get("beam_width", 8))
    pool_size = pool_size or max(6, beam_width * 3)
    min_distance = int(config.get("min_site_distance", 3))
    penalty_coefficient = float(config.get("epitasis_penalty", 0.35))

    pool = singles[:pool_size]
    if len(pool) < 2:
        return []

    # ---------- 阶段一：束搜索（廉价估计） ----------
    # 每条 beam 是一个组合；从单点开始逐位扩展
    beams: list[CombinationCandidate] = [
        CombinationCandidate(mutations=[item], estimated_score=item.score) for item in pool
    ]
    beams.sort(key=lambda item: -item.estimated_score)
    beams = beams[:beam_width]

    completed: list[CombinationCandidate] = []

    for _ in range(2, max_sites + 1):
        expanded: list[CombinationCandidate] = []
        for beam in beams:
            used = {item.position for item in beam.mutations}
            for candidate in pool:
                if candidate.position in used:
                    continue
                merged = [*beam.mutations, candidate]
                positions = [item.position for item in merged]
                penalty = _epitasis_penalty(positions, min_distance, penalty_coefficient)
                # 估计分 = 单点分之和 - 上位惩罚；用均摊方式避免组合越长分越高
                raw_sum = sum(item.score for item in merged)
                estimated = raw_sum / len(merged) - penalty
                expanded.append(
                    CombinationCandidate(
                        mutations=merged,
                        estimated_score=estimated,
                        epitasis_penalty=penalty,
                        details={"raw_sum": round(raw_sum, 2), "site_count": len(merged)},
                    )
                )

        # 去重（同一组位点只保留最优）
        best_by_key: dict[tuple[int, ...], CombinationCandidate] = {}
        for item in expanded:
            key = tuple(sorted(item.positions))
            if key not in best_by_key or item.estimated_score > best_by_key[key].estimated_score:
                best_by_key[key] = item

        ranked = sorted(best_by_key.values(), key=lambda item: -item.estimated_score)
        completed.extend(ranked[: beam_width * 2])
        beams = ranked[:beam_width]
        if not beams:
            break

    # 加上单点本身（组合搜索不应丢掉最好的单点方案）
    all_candidates = completed + [
        CombinationCandidate(mutations=[item], estimated_score=item.score) for item in pool[:beam_width]
    ]
    all_candidates.sort(key=lambda item: -item.estimated_score)

    # 去重后再排序
    seen: set[tuple[int, ...]] = set()
    unique: list[CombinationCandidate] = []
    for item in all_candidates:
        key = tuple(sorted(item.positions))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    # ---------- 阶段二：对头部候选做上下文依赖精评 ----------
    if refine_fn is not None:
        for item in unique[:refine_top_k]:
            mutations = {entry.position: entry.mutant for entry in item.mutations}
            try:
                item.refined_score = float(refine_fn(mutations, item))
                item.refined = True
            except Exception as exc:  # 精评失败不应中断整个搜索
                logger.warning("组合精评失败 %s: %s", item.labels, exc)

    # 排序：优先用精评分，未精评的按估计分
    unique.sort(key=lambda item: -item.final_score)
    logger.info(
        "组合搜索完成：池=%d 候选=%d 精评=%d",
        len(pool),
        len(unique),
        sum(1 for item in unique if item.refined),
    )
    return unique


def combination_sequence(sequence: str, mutations: dict[int, str]) -> str:
    """构造组合突变序列（薄封装，便于统一调用点）。"""
    return build_combinatorial_sequence(sequence, mutations)
