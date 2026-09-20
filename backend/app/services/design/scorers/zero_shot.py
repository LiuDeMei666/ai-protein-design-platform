"""零样本突变打分（ESM-2 掩码边缘似然比）。

核心公式::

    ΔlogP(a -> b) = logP(b | context\\{a}) - logP(a | context\\{a})

其中 ``context\\{a}`` 表示把位置 ``a`` 掩码后的上下文。该值出自蛋白语言模型对
"这个位置能不能换成 b"的判断，**不需要任何标注数据**，因此在企业实验数据
到位前即可给出有意义的排序。

本模块负责：
1. 调用 ESM-2 拿到 ΔlogP 矩阵；
2. 把 ΔlogP 归一到 0-100 的"稳定性贡献"分；
3. 提供"保守性/活性影响"分（同一模型信号的功能侧解读）。

注意：稳定性贡献与活性影响用的是**同一个模型的同一份信号**，但解读不同——
前者关心替换是否被模型接受（越接近 0 越好），后者关心该位点本身是否高度保守
（野生型对数概率越低 = 位点越不确定 = 越可能容忍替换）。
两者相关性高但不等价，报告中会明确说明这一点，避免用户以为是两个独立证据。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ....core.config import load_platform_config
from ....core.logging import describe_sequence, get_logger
from ...embedding.esm2 import AA_ORDER, get_service

logger = get_logger(__name__)


@dataclass
class ZeroShotScores:
    """全位点零样本打分结果。"""

    positions: list[int]
    aa_order: str
    #: ΔlogP 矩阵 [n_positions, 20]
    delta_logprob: np.ndarray
    #: 野生型残基的掩码对数概率 [n_positions]
    wt_logprob: np.ndarray
    #: 归一到 0-100 的稳定性贡献分 [n_positions, 20]
    stability: np.ndarray
    #: 窗口信息（超长序列分窗口时非空）
    windows: list[tuple[int, int]]

    def index_of(self, position: int) -> int | None:
        try:
            return self.positions.index(position)
        except ValueError:
            return None

    def delta(self, position: int, mutant: str) -> float | None:
        """查询某个突变的 ΔlogP。"""
        row = self.index_of(position)
        if row is None or mutant not in self.aa_order:
            return None
        return float(self.delta_logprob[row, self.aa_order.index(mutant)])

    def stability_score(self, position: int, mutant: str) -> float | None:
        """查询某个突变的稳定性贡献分（0-100）。"""
        row = self.index_of(position)
        if row is None or mutant not in self.aa_order:
            return None
        return float(self.stability[row, self.aa_order.index(mutant)])

    def wt_logp(self, position: int) -> float | None:
        row = self.index_of(position)
        if row is None:
            return None
        value = float(self.wt_logprob[row])
        return None if np.isnan(value) else value


def _mapping() -> tuple[float, float]:
    config = load_platform_config().get("design", {}).get("zero_shot", {})
    return float(config.get("delta_worst", -7.0)), float(config.get("delta_best", 0.0))


def normalize_delta(delta: np.ndarray) -> np.ndarray:
    """把 ΔlogP 线性映射到 0-100。"""
    worst, best = _mapping()
    if best == worst:
        return np.full_like(delta, 50.0, dtype=np.float64)
    values = (delta - worst) / (best - worst) * 100.0
    return np.clip(values, 0.0, 100.0).astype(np.float32)


def compute_zero_shot(
    sequence: str,
    positions: list[int] | None = None,
    batch_size: int | None = None,
) -> ZeroShotScores:
    """计算全位点 × 20 氨基酸的零样本打分。

    实现要点：**批量掩码前向**。对 L 个位点做 L 次掩码，按 ``batch_size`` 分组，
    一次前向处理整组，因此总代价约为 ``L / batch_size`` 次前向，而不是 L 次。
    """
    service = get_service()
    service.ensure_loaded()

    target = positions if positions is not None else list(range(len(sequence)))
    result = service.masked_marginal(sequence, positions=target, batch_size=batch_size)

    logger.info(
        "零样本打分 %s 位点=%d 窗口=%d",
        describe_sequence(sequence),
        len(result.positions),
        len(result.windows),
    )

    return ZeroShotScores(
        positions=result.positions,
        aa_order=AA_ORDER,
        delta_logprob=result.delta_logprob,
        wt_logprob=result.wt_logprob,
        stability=normalize_delta(result.delta_logprob),
        windows=result.windows,
    )


def conservation_level(wt_logprob: float | None, threshold: float | None = None) -> tuple[float, str]:
    """由野生型掩码对数概率判断位点保守程度。

    返回值：(0-100 的"可突变空间"分, 中文描述)。
    野生型对数概率越接近 0，说明上下文对该残基的确定度越高 -> 越保守 -> 可突变空间越小。
    """
    if threshold is None:
        threshold = float(
            load_platform_config().get("scan", {}).get("conserve_threshold", 0.85)
        )
    if wt_logprob is None or np.isnan(wt_logprob):
        return 50.0, "无法判定保守性（模型未返回该位点）"

    # -2.0 以上视为高度保守，-6.0 以下视为非常宽松
    if wt_logprob >= -0.5:
        return 8.0, "极高保守：上下文几乎唯一确定该残基，替换风险极高"
    if wt_logprob >= -1.5:
        return 25.0, "高度保守：该残基受强上下文约束，替换需实验验证"
    if wt_logprob >= -3.0:
        return 55.0, "中度保守：存在一定替换空间"
    if wt_logprob >= -5.0:
        return 78.0, "较宽松：该位点受约束较弱，替换相对可容忍"
    return 92.0, "宽松：上下文对该残基约束很弱（常见于柔性/表面区）"


def activity_score(
    delta_logp: float | None,
    wt_logprob: float | None,
    proximity_factor: float = 1.0,
    rulepack_factor: float = 1.0,
) -> tuple[float, str]:
    """活性影响分（0-100，越高表示对功能影响越小）。

    Args:
        delta_logp: 该突变的 ΔlogP。
        wt_logprob: 该位点的野生型掩码对数概率。
        proximity_factor: 与功能位点距离的折扣因子（0-1，越近越小）。
        rulepack_factor: 规则包给出的活性偏好因子（0-1，越大越有利）。
    """
    config = load_platform_config().get("design", {}).get("activity", {})
    w_conservation = float(config.get("conservation_weight", 0.55))
    w_proximity = float(config.get("proximity_weight", 0.25))
    w_rulepack = float(config.get("rulepack_weight", 0.20))

    worst, best = _mapping()
    if delta_logp is None:
        conservation_score = 50.0
    else:
        span = best - worst
        conservation_score = float(np.clip((delta_logp - worst) / span * 100.0, 0.0, 100.0))

    proximity_score = float(np.clip(proximity_factor, 0.0, 1.0)) * 100.0
    rulepack_score = float(np.clip(rulepack_factor, 0.0, 1.0)) * 100.0

    total = (
        w_conservation * conservation_score
        + w_proximity * proximity_score
        + w_rulepack * rulepack_score
    ) / max(1e-6, w_conservation + w_proximity + w_rulepack)

    description = (
        f"模型对替换{'接受度较高' if conservation_score >= 60 else '接受度偏低'}"
        f"（ΔlogP={delta_logp:.2f}）" if delta_logp is not None else "模型未返回该位点信息"
    )
    if proximity_factor < 0.6:
        description += "；该位点邻近已知功能位点，需谨慎"
    if rulepack_factor < 0.6:
        description += "；不符合该蛋白类型的活性改造偏好"

    return round(float(total), 2), description


def best_substitutions(scores: ZeroShotScores, position: int, top_n: int = 5) -> list[tuple[str, float, float]]:
    """返回某位点上模型最偏好的若干替换：[(氨基酸, ΔlogP, 稳定性分), ...]。"""
    row = scores.index_of(position)
    if row is None:
        return []
    wild_type = None
    order = np.argsort(-scores.delta_logprob[row])
    results: list[tuple[str, float, float]] = []
    for column in order:
        amino_acid = scores.aa_order[int(column)]
        results.append(
            (
                amino_acid,
                float(scores.delta_logprob[row, column]),
                float(scores.stability[row, column]),
            )
        )
        if len(results) >= top_n:
            break
    del wild_type  # 保留接口对称性
    return results


def summarize_scores(scores: ZeroShotScores) -> dict[str, Any]:
    """打分矩阵的统计摘要（写入作业结果，便于审计）。"""
    return {
        "position_count": len(scores.positions),
        "aa_order": scores.aa_order,
        "delta_logprob_min": round(float(np.nanmin(scores.delta_logprob)), 3) if scores.delta_logprob.size else None,
        "delta_logprob_mean": round(float(np.nanmean(scores.delta_logprob)), 3) if scores.delta_logprob.size else None,
        "wt_logprob_mean": round(float(np.nanmean(scores.wt_logprob)), 3) if scores.wt_logprob.size else None,
        "windows": [list(window) for window in scores.windows],
    }
