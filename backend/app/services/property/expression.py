"""表达量趋势预测（宿主表达潜力）。

重要说明：只有蛋白序列时能算什么
--------------------------------
拿到的是**氨基酸序列**而不是基因序列，因此无法直接计算密码子适应指数（CAI）——
CAI 需要实际的 DNA 密码子。本项目诚实地区分两种模式：

1. **蛋白序列模式**（当前）：使用可以从氨基酸序列严格推导的特征——
   * **宿主氨基酸使用偏好距离**：从 Kazusa 密码子表汇总出该宿主蛋白组的氨基酸
     使用频率，与目标蛋白组成做余弦/卡方距离。距离越大说明该蛋白对宿主的
     tRNA/氨酰-tRNA 合成酶池构成额外负担。
   * **密码子优化空间（headroom）**：某个氨基酸在该宿主中最优密码子的使用频率。
     若最优密码子本身就低频，则该残基的可优化空间有限。
   * **N 端规则**（Bachmair/Varshavsky）：Met 之后第二个残基决定半衰期。
   * **N 端疏水段**：易被导向膜或形成包涵体。
   * **精氨酸双联体等核糖体停滞基序**（E. coli 中 Arg-Arg 是已知的停滞热点）。
   * 低复杂度区域导致的翻译停滞。

2. **基因序列模式**（预留）：若企业提供 DNA 序列，调用 :func:`compute_cai`
   给出标准 CAI（Sharp & Li 1987）。

宿主密码子表来自 ``data/seeds/codon_usage/*.json``（Kazusa 官方数据，
``scripts/fetch_codon_usage.py`` 生成），并记录参考集规模供判断可信度。
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from typing import Any

from ...core.config import get_settings, load_platform_config
from ...core.logging import get_logger
from ..sequence.feature_utils import KYTE_DOOLITTLE, low_complexity_score
from .biophys import N_END_RULE
from .metric import Evidence, Metric, aggregate_weighted, linear_score

logger = get_logger(__name__)

#: 核糖体停滞敏感基序（E. coli 中经实验证实）
STALLING_MOTIFS: dict[str, float] = {
    "RR": 1.0,
    "RPR": 0.7,
    "PPP": 0.8,
    "GGG": 0.6,
    "KKK": 0.5,
}


@functools.lru_cache(maxsize=16)
def load_codon_table(table_name: str) -> dict[str, Any] | None:
    """加载密码子使用表（带缓存）。找不到时返回 ``None``。"""
    settings = get_settings()
    path = Path(settings.seeds_dir) / "codon_usage" / f"{table_name}.json"
    if not path.exists():
        logger.warning(
            "密码子表不存在: %s，请先执行 python scripts/fetch_codon_usage.py", path
        )
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("读取密码子表失败 %s: %s", path, exc)
        return None


def resolve_host(host_system: str) -> tuple[str, dict[str, Any] | None]:
    """把宿主名解析为 (显示名, 密码子表)。"""
    config = load_platform_config().get("host_systems", {})
    entry = config.get(host_system) or config.get("ecoli") or {}
    label = entry.get("label", host_system)
    table_name = entry.get("table", "ecoli_k12")
    return label, load_codon_table(table_name)


def host_amino_acid_usage(table: dict[str, Any]) -> dict[str, float]:
    """由密码子表汇总宿主蛋白组的氨基酸使用频率（每千残基）。"""
    usage: dict[str, float] = {}
    for record in table.get("codons", {}).values():
        amino_acid = record.get("amino_acid", "")
        if not amino_acid or amino_acid == "*":
            continue
        usage[amino_acid] = usage.get(amino_acid, 0.0) + float(record.get("per_thousand", 0.0))
    return usage


def codon_optimization_headroom(
    sequence: str, table: dict[str, Any]
) -> tuple[float, dict[str, float]]:
    """平均"最优密码子可用度"。

    对每种氨基酸取其在本宿主中频率最高的密码子的 ``fraction``，
    再对序列取平均。值接近 1 表示每个残基都有高使用率的密码子可用；
    值偏低说明该蛋白含有宿主本身就不偏好使用的氨基酸。
    """
    best: dict[str, float] = {}
    for record in table.get("codons", {}).values():
        amino_acid = record.get("amino_acid", "")
        if not amino_acid or amino_acid == "*":
            continue
        best[amino_acid] = max(best.get(amino_acid, 0.0), float(record.get("fraction", 0.0)))

    values = [best[char] for char in sequence if char in best]
    if not values:
        return 0.0, best
    return sum(values) / len(values), best


def composition_distance(sequence: str, host_usage: dict[str, float]) -> float:
    """目标蛋白组成与宿主蛋白组组成的卡方距离（归一化后取平均）。"""
    counts: dict[str, int] = {}
    for char in sequence:
        counts[char] = counts.get(char, 0) + 1
    length = max(1, len(sequence))

    total_host = sum(host_usage.values()) or 1.0
    distance = 0.0
    considered = 0
    for amino_acid, host_value in host_usage.items():
        observed = counts.get(amino_acid, 0) / length
        expected = host_value / total_host
        if expected <= 0:
            continue
        distance += (observed - expected) ** 2 / expected
        considered += 1
    return distance / considered if considered else 0.0


def count_stalling_motifs(sequence: str) -> tuple[float, list[dict[str, Any]]]:
    """统计核糖体停滞敏感基序（加权计数）。"""
    total = 0.0
    loci: list[dict[str, Any]] = []
    for motif, weight in STALLING_MOTIFS.items():
        start = 0
        while True:
            index = sequence.find(motif, start)
            if index < 0:
                break
            total += weight
            loci.append(
                {
                    "position": index,
                    "residue": sequence[index],
                    "motif": motif,
                    "type": "stalling_motif",
                    "severity": round(weight, 2),
                }
            )
            start = index + 1
    return total, loci


def n_terminal_hydrophobic_run(sequence: str, length: int = 18) -> float:
    """N 端前 ``length`` 个残基的平均疏水性（信号肽/膜靶向风险）。"""
    if not sequence:
        return 0.0
    window = sequence[:length]
    return sum(KYTE_DOOLITTLE.get(char, 0.0) for char in window) / len(window)


def compute_cai(dna_sequence: str, table: dict[str, Any]) -> float:
    """标准密码子适应指数（Sharp & Li 1987）。

    ``w = RSCU(codon) / RSCU(该氨基酸最常用密码子)``，``CAI = exp(mean(ln w))``。
    Kazusa 表中的 ``fraction`` 即为该氨基酸家族内的 RSCU，可直接使用。
    """
    fractions: dict[str, float] = {}
    for codon, record in table.get("codons", {}).items():
        fractions[codon] = float(record.get("fraction", 0.0))

    best: dict[str, float] = {}
    for codon, record in table.get("codons", {}).items():
        amino_acid = record.get("amino_acid", "")
        if not amino_acid or amino_acid == "*":
            continue
        best[amino_acid] = max(best.get(amino_acid, 0.0), fractions.get(codon, 0.0))

    log_sum = 0.0
    count = 0
    sequence = dna_sequence.upper().replace("U", "T")
    for index in range(0, len(sequence) - 2, 3):
        codon = sequence[index : index + 3]
        record = table.get("codons", {}).get(codon)
        if record is None:
            continue
        amino_acid = record.get("amino_acid", "")
        if not amino_acid or amino_acid == "*":
            continue
        denominator = best.get(amino_acid, 0.0)
        if denominator <= 0:
            continue
        ratio = max(fractions.get(codon, 0.0), 1e-6) / denominator
        log_sum += math.log(ratio)
        count += 1

    return round(math.exp(log_sum / count), 4) if count else 0.0


def assess_expression(
    sequence: str,
    biophys_values: dict[str, Any],
    host_system: str = "ecoli",
    dna_sequence: str | None = None,
) -> Metric:
    """表达量趋势评估：分数越高表示在宿主中越可能高产可溶表达。"""
    length = max(1, len(sequence))
    label, table = resolve_host(host_system)
    counts = biophys_values.get("counts", {})

    second_residue = sequence[1] if len(sequence) > 1 else ""
    n_end_class = N_END_RULE.get(second_residue, "unknown")
    n_end_score_map = {"stabilizing": 92.0, "neutral": 68.0, "destabilizing": 38.0, "unknown": 55.0}

    n_terminal_hydro = n_terminal_hydrophobic_run(sequence)
    stalling_score, stalling_loci = count_stalling_motifs(sequence)
    low_complexity = low_complexity_score(sequence)
    cys_trp = (counts.get("C", 0) + counts.get("W", 0)) / length

    headroom: float | None = None
    distance: float | None = None
    table_note = "密码子表不可用"
    if table is not None:
        headroom, _ = codon_optimization_headroom(sequence, table)
        distance = composition_distance(sequence, host_amino_acid_usage(table))
        table_note = (
            f"{table.get('organism', host_system)}，"
            f"{table.get('n_cds', 0)} CDS / {table.get('n_codons', 0)} codons"
        )

    cai_value: float | None = None
    if dna_sequence and table is not None:
        cai_value = compute_cai(dna_sequence, table)

    parts = [
        (
            0.26,
            headroom * 100.0 if headroom is not None else None,
            Evidence(
                label="最优密码子可用度",
                value=round(headroom, 4) if headroom is not None else "不可用",
                rationale=(
                    "每种氨基酸在宿主中最常用密码子的使用频率之均值。"
                    f"参考表：{table_note}。"
                ),
            ),
        ),
        (
            0.20,
            linear_score(distance, worst=0.020, best=0.002) if distance is not None else None,
            Evidence(
                label="宿主氨基酸使用偏好距离（卡方）",
                value=round(distance, 5) if distance is not None else "不可用",
                rationale="目标蛋白组成偏离宿主蛋白组越远，对 tRNA 池与合成酶的额外负担越大。",
            ),
        ),
        (
            0.16,
            n_end_score_map.get(n_end_class, 55.0),
            Evidence(
                label="N 端规则（Met 后第二位残基）",
                value=f"{second_residue or 'NA'}（{n_end_class}）",
                rationale=(
                    "N 端规则：Met 后为 R/K/F/L/W/Y/I/D/E/N/Q 时易被蛋白酶体/胞内蛋白酶识别降解，"
                    "为 A/C/G/P/S/T/V 时半衰期显著更长。"
                ),
            ),
        ),
        (
            0.14,
            linear_score(n_terminal_hydro, worst=1.9, best=0.2),
            Evidence(
                label="N 端 18 残基平均疏水性",
                value=round(n_terminal_hydro, 3),
                rationale="N 端强疏水段易被信号识别颗粒捕获并导向膜，或直接形成包涵体。",
            ),
        ),
        (
            0.12,
            linear_score(stalling_score, worst=max(3.0, length * 0.03), best=0.0),
            Evidence(
                label="核糖体停滞敏感基序加权计数",
                value=round(stalling_score, 2),
                rationale="Arg-Arg 等双联体在大肠杆菌中引起核糖体停滞与翻译中断（+1 移码热区）。",
            ),
        ),
        (
            0.07,
            linear_score(cys_trp, worst=0.075, best=0.015),
            Evidence(
                label="Cys + Trp 占比",
                value=round(cys_trp, 4),
                rationale="含硫/吲哚残基的合成与正确折叠成本高，占比过高通常降低表达产量。",
            ),
        ),
        (
            0.05,
            linear_score(low_complexity, worst=0.35, best=0.0),
            Evidence(
                label="低复杂度区域占比",
                value=round(low_complexity, 4),
                rationale="低复杂度序列易发生翻译停滞、mRNA 二级结构异常与聚集。",
            ),
        ),
        (
            0.10,
            cai_value * 100.0 if cai_value is not None else None,
            Evidence(
                label="密码子适应指数（CAI，需 DNA 序列）",
                value=round(cai_value, 4) if cai_value is not None else "不可用（仅提供蛋白序列）",
                rationale="提供基因序列时按 Sharp & Li 1987 计算标准 CAI；仅蛋白序列时无法计算。",
            ),
        ),
    ]

    score, evidence = aggregate_weighted(parts)

    return Metric(
        key="expression",
        label="表达量趋势",
        score=round(score, 1),
        algorithm=(
            "宿主密码子使用偏好（Kazusa 官方表）+ N 端规则 + N 端疏水段"
            " + 核糖体停滞基序 + 含硫残基负荷"
        ),
        rationale=(
            f"宿主体系：{label}。分数越高表示越可能在该宿主中获得高产可溶表达。"
            "注意：仅凭氨基酸序列无法计算真正的 CAI；本项给出的是可严格推导的代理特征。"
            + ("若提供基因序列，将直接使用 CAI。" if not dna_sequence else "")
        ),
        confidence=0.5 if headroom is not None else 0.35,
        evidence=evidence,
        locus=stalling_loci[:200],
        meta={
            "host_system": host_system,
            "host_label": label,
            "codon_table": table_note,
            "n_end_rule_class": n_end_class,
            "headroom": round(headroom, 4) if headroom is not None else None,
            "composition_distance": round(distance, 5) if distance is not None else None,
            "cai": cai_value,
        },
    )
