"""溶解性预测。

算法依据
--------
重组表达中可溶性主要受以下已发表特征支配（Wilkinson & Harrison 1991；
Davis et al. 1999；Chan et al. 2013 的可溶性标签研究）：

1. **转角形成残基占比**（S/T/N/Q/G/P）：占比高 → 表面更亲水、可溶性更好。
2. **平均净电荷密度**：表面电荷高有利于静电排斥、抑制聚集。
3. **半胱氨酸数量**：非配对游离 Cys 易形成错配二硫键与共价聚集。
4. **GRAVY 疏水指数**：越低越亲水。
5. **最长连续疏水片段**：过长疏水段是包涵体形成的强预测因子。
6. **低复杂度区域**：与表达困难、聚集相关。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...core.config import load_platform_config
from ...core.logging import get_logger
from ..sequence.feature_utils import KYTE_DOOLITTLE, HYDROPHOBIC_AA, net_charge
from .metric import Evidence, Metric, aggregate_weighted, linear_score

logger = get_logger(__name__)

TURN_FORMING = frozenset("STNQGP")


@dataclass
class HydrophobicRun:
    """一段连续疏水区域。"""

    start: int
    end: int
    length: int
    mean_hydrophobicity: float


def find_hydrophobic_runs(sequence: str, min_length: int = 5, window: int = 5) -> list[HydrophobicRun]:
    """找出连续疏水片段（滑窗均值 > 1.6，即明显疏水）。"""
    runs: list[HydrophobicRun] = []
    current_start: int | None = None
    length = len(sequence)

    for start in range(0, max(0, length - window + 1)):
        segment = sequence[start : start + window]
        mean = sum(KYTE_DOOLITTLE.get(char, 0.0) for char in segment) / window
        is_hydrophobic = mean > 1.6
        if is_hydrophobic and current_start is None:
            current_start = start
        elif not is_hydrophobic and current_start is not None:
            end = start + window - 1
            if end - current_start >= min_length:
                runs.append(_build_run(sequence, current_start, end))
            current_start = None

    if current_start is not None:
        end = length
        if end - current_start >= min_length:
            runs.append(_build_run(sequence, current_start, end))

    return runs


def _build_run(sequence: str, start: int, end: int) -> HydrophobicRun:
    segment = sequence[start:end]
    mean = sum(KYTE_DOOLITTLE.get(char, 0.0) for char in segment) / max(1, len(segment))
    return HydrophobicRun(start=start, end=end, length=end - start, mean_hydrophobicity=round(mean, 3))


def assess_solubility(sequence: str, biophys_values: dict[str, Any]) -> Metric:
    """可溶性评估，返回 0-100 分（越高越可溶）。"""
    length = max(1, len(sequence))
    counts = biophys_values.get("counts", {})
    gravy = float(biophys_values.get("gravy", 0.0))
    charge = abs(float(biophys_values.get("net_charge_ph7", 0.0))) / length

    turn_fraction = sum(counts.get(aa, 0) for aa in TURN_FORMING) / length
    cys_fraction = counts.get("C", 0) / length
    hydrophobic_runs = find_hydrophobic_runs(sequence)
    longest_run = max((run.length for run in hydrophobic_runs), default=0)

    config = load_platform_config()
    low_complexity_cutoff = float(config.get("structure", {}).get("burial_sasa_cutoff", 0.25))

    parts = [
        (
            0.24,
            linear_score(turn_fraction, worst=0.20, best=0.42),
            Evidence(
                label="转角形成残基占比（STNQGP）",
                value=round(turn_fraction, 4),
                rationale="转角残基富集于蛋白表面且高度亲水，是经典的可溶性正向预测因子。",
            ),
        ),
        (
            0.20,
            linear_score(charge, worst=0.02, best=0.16),
            Evidence(
                label="净电荷密度（|净电荷|/长度，pH 7）",
                value=round(charge, 4),
                rationale="表面净电荷通过静电排斥抑制分子间聚集，是包涵体形成的最强负相关特征之一。",
            ),
        ),
        (
            0.16,
            linear_score(gravy, worst=0.15, best=-0.55),
            Evidence(
                label="GRAVY 疏水指数",
                value=round(gravy, 4),
                rationale="GRAVY 越低表示整体越亲水，可溶性通常越好。",
            ),
        ),
        (
            0.16,
            linear_score(float(longest_run), worst=14.0, best=3.0),
            Evidence(
                label="最长连续疏水片段长度",
                value=longest_run,
                rationale="超过 10 个残基的连续疏水段（滑窗 Kyte-Doolittle 均值 > 1.6）是包涵体形成的高危结构。",
            ),
        ),
        (
            0.14,
            linear_score(cys_fraction, worst=0.035, best=0.0),
            Evidence(
                label="半胱氨酸占比",
                value=round(cys_fraction, 4),
                rationale="游离 Cys 易发生氧化与二硫键错配，导致共价聚集与可溶性下降。",
            ),
        ),
        (
            0.10,
            linear_score(float(len(hydrophobic_runs)), worst=8.0, best=0.0),
            Evidence(
                label="疏水片段数量",
                value=len(hydrophobic_runs),
                rationale="多个分散疏水段比单一疏水段更易在折叠中间体暴露，引发聚集。",
            ),
        ),
    ]

    score, evidence = aggregate_weighted(parts)
    loci = [
        {
            "position": run.start,
            "end": run.end,
            "length": run.length,
            "type": "hydrophobic_run",
            "severity": round(min(1.0, run.length / 15.0), 2),
            "sequence": sequence[run.start : run.end],
        }
        for run in hydrophobic_runs
    ]

    return Metric(
        key="solubility",
        label="溶解性",
        score=round(score, 1),
        algorithm="转角残基占比 + 净电荷密度 + GRAVY + 最长疏水片段 + Cys 负荷",
        rationale=(
            "以重组表达可溶性的已发表序列决定因素评分。"
            "若分数偏低，优先考虑把疏水段中暴露的疏水残基改为带电/极性残基（如 V→E/K）。"
        ),
        confidence=0.55,
        evidence=evidence,
        locus=loci[:200],
        meta={
            "longest_hydrophobic_run": longest_run,
            "hydrophobic_run_count": len(hydrophobic_runs),
            "low_complexity_cutoff": low_complexity_cutoff,
        },
    )
