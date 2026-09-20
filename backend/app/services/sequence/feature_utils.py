"""序列特征工具：组成、滑窗、疏水矩、胶原 Gly-X-Y 识别。

这些是**纯序列**层面的确定性计算，不依赖任何模型，因此结果完全可复现，
在报告中作为"结构无关"的证据链。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

STANDARD_AA: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle 疏水性标度（1982）
KYTE_DOOLITTLE: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Eisenberg 共识疏水性标度，用于疏水矩计算
EISENBERG: dict[str, float] = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}

# Chou-Fasman 二级结构倾向
HELIX_PROPENSITY: dict[str, float] = {
    "A": 1.42, "C": 0.70, "D": 1.01, "E": 1.51, "F": 1.13, "G": 0.57,
    "H": 1.00, "I": 1.08, "K": 1.16, "L": 1.21, "M": 1.45, "N": 0.67,
    "P": 0.57, "Q": 1.11, "R": 0.98, "S": 0.77, "T": 0.83, "V": 1.06,
    "W": 1.08, "Y": 0.69,
}
SHEET_PROPENSITY: dict[str, float] = {
    "A": 0.83, "C": 1.19, "D": 0.54, "E": 0.37, "F": 1.38, "G": 0.75,
    "H": 0.87, "I": 1.60, "K": 0.74, "L": 1.30, "M": 1.05, "N": 0.89,
    "P": 0.55, "Q": 1.10, "R": 0.93, "S": 0.75, "T": 1.19, "V": 1.70,
    "W": 1.37, "Y": 1.47,
}
TURN_PROPENSITY: dict[str, float] = {
    "A": 0.66, "C": 1.19, "D": 1.46, "E": 0.74, "F": 0.60, "G": 1.56,
    "H": 0.95, "I": 0.47, "K": 1.01, "L": 0.59, "M": 0.60, "N": 1.56,
    "P": 1.52, "Q": 0.98, "R": 0.95, "S": 1.43, "T": 0.96, "V": 0.50,
    "W": 0.96, "Y": 1.14,
}

# 带电残基
POSITIVE_AA = frozenset("KRH")
NEGATIVE_AA = frozenset("DE")
AROMATIC_AA = frozenset("FWY")
ALIPHATIC_AA = frozenset("AILV")
HYDROPHOBIC_AA = frozenset("AVILMFWC")
POLAR_AA = frozenset("STNQY")
FLEXIBLE_AA = frozenset("GSPTQ")


def composition(sequence: str) -> dict[str, float]:
    """氨基酸组成（占比，和为 1）。"""
    if not sequence:
        return {aa: 0.0 for aa in STANDARD_AA}
    length = len(sequence)
    counts: dict[str, int] = {aa: 0 for aa in STANDARD_AA}
    for char in sequence:
        if char in counts:
            counts[char] += 1
    return {aa: counts[aa] / length for aa in STANDARD_AA}


def composition_counts(sequence: str) -> dict[str, int]:
    """氨基酸计数。"""
    counts: dict[str, int] = {aa: 0 for aa in STANDARD_AA}
    for char in sequence:
        if char in counts:
            counts[char] += 1
    return counts


def window_slices(length: int, window: int, step: int = 1) -> Iterable[tuple[int, int]]:
    """滑窗区间生成器（半开区间）。"""
    if window <= 0 or length < window:
        return
    for start in range(0, length - window + 1, step):
        yield start, start + window


def sliding_hydrophobicity(sequence: str, window: int = 9, step: int = 1) -> list[float]:
    """滑窗平均疏水性（Kyte-Doolittle）。"""
    scores: list[float] = []
    for start, end in window_slices(len(sequence), window, step):
        segment = sequence[start:end]
        values = [KYTE_DOOLITTLE.get(char, 0.0) for char in segment]
        scores.append(sum(values) / len(values))
    return scores


def hydrophobic_moment(sequence: str, window: int = 11, angle: float = 100.0) -> float:
    """疏水矩（Eisenberg）：衡量序列形成两亲性螺旋的倾向。

    值越高越可能形成可被免疫系统识别的两亲性表面，用于免疫原性启发式。
    """
    import math

    if len(sequence) < window:
        return 0.0

    best = 0.0
    delta = math.radians(angle)
    for start, end in window_slices(len(sequence), window, 1):
        sin_sum = 0.0
        cos_sum = 0.0
        for offset, char in enumerate(sequence[start:end]):
            value = EISENBERG.get(char, 0.0)
            sin_sum += value * math.sin(offset * delta)
            cos_sum += value * math.cos(offset * delta)
        best = max(best, math.hypot(sin_sum, cos_sum) / window)
    return best


def is_gly_xy_repeat(sequence: str, position: int) -> bool:
    """判断 ``position`` 处是否处于胶原 Gly-X-Y 重复的 Gly 位。"""
    if position < 0 or position >= len(sequence):
        return False
    if sequence[position] != "G":
        return False
    # 向前回溯：该位置是否与 (±3k) 上的 G 形成重复
    hits = 0
    for offset in (-9, -6, -3, 3, 6, 9):
        neighbour = position + offset
        if 0 <= neighbour < len(sequence) and sequence[neighbour] == "G":
            hits += 1
    return hits >= 2


def gly_xy_phase(sequence: str, position: int) -> int | None:
    """返回残基在 Gly-X-Y 三联体中的相位（0=Gly, 1=X, 2=Y）。

    以最近的 Gly 锚点推断；无法判定时返回 ``None``。
    """
    if not sequence:
        return None
    for anchor in (position, position - 1, position - 2):
        if anchor < 0:
            continue
        if sequence[anchor] == "G" and (anchor == 0 or is_gly_xy_repeat(sequence, anchor)):
            return (position - anchor) % 3
    return None


def collagen_repeat_density(sequence: str, window: int = 30) -> float:
    """Gly-X-Y 重复密度：窗口中 Gly 占比接近 1/3 且周期性出现时的比例。"""
    if len(sequence) < window:
        window = len(sequence)
    if window < 9:
        return 0.0

    hit_windows = 0
    total_windows = 0
    for start, end in window_slices(len(sequence), window, step=max(1, window // 3)):
        segment = sequence[start:end]
        total_windows += 1
        gly_positions = [index for index, char in enumerate(segment) if char == "G"]
        if not gly_positions:
            continue
        # 检查主周期是否为 3
        spacings = [
            gly_positions[index + 1] - gly_positions[index]
            for index in range(len(gly_positions) - 1)
        ]
        if not spacings:
            continue
        in_phase = sum(1 for gap in spacings if gap % 3 == 0)
        if in_phase / len(spacings) >= 0.6:
            hit_windows += 1
    return hit_windows / total_windows if total_windows else 0.0


def low_complexity_score(sequence: str, window: int = 20) -> float:
    """低复杂度区域占比：单一残基占比超过 50% 的窗口比例。"""
    if len(sequence) < window:
        return 0.0
    flagged = 0
    total = 0
    for start, end in window_slices(len(sequence), window, step=window // 2 or 1):
        segment = sequence[start:end]
        total += 1
        counts: dict[str, int] = {}
        for char in segment:
            counts[char] = counts.get(char, 0) + 1
        if max(counts.values()) / len(segment) > 0.5:
            flagged += 1
    return flagged / total if total else 0.0


def net_charge(sequence: str, ph: float = 7.0) -> float:
    """指定 pH 下的净电荷（Henderson-Hasselbalch，简化 pKa 集）。"""
    pka_positive = {"K": 10.5, "R": 12.4, "H": 6.0, "N_term": 9.6}
    pka_negative = {"D": 3.9, "E": 4.3, "C": 8.3, "Y": 10.1, "C_term": 2.4}

    charge = 0.0
    for residue, pka in pka_positive.items():
        if residue == "N_term":
            charge += 1.0 / (1.0 + 10 ** (ph - pka))
        else:
            count = sequence.count(residue)
            if count:
                charge += count / (1.0 + 10 ** (ph - pka))
    for residue, pka in pka_negative.items():
        if residue == "C_term":
            charge -= 1.0 / (1.0 + 10 ** (pka - ph))
        else:
            count = sequence.count(residue)
            if count:
                charge -= count / (1.0 + 10 ** (pka - ph))
    return charge


#: 信号肽疏水核心（h-region）允许的残基
SIGNAL_HYDROPHOBIC = frozenset("AVLIFWMC")


def detect_signal_peptide(sequence: str, max_scan: int = 50) -> tuple[int, int] | None:
    """启发式识别 N 端信号肽，返回 ``(0, end)`` 区间或 ``None``。

    为什么需要它
    ------------
    分泌型重组蛋白（胶原蛋白、工业酶、蛋白 A）的 N 端都有信号肽，
    **在成熟过程中被切除**。若平台在这些位点推荐突变，对最终产品毫无意义——
    这正是实测中遇到的问题：全长 COL1A1 的 Top 候选落在第 2 位、
    蛋白 A 落在第 5 位，两者都位于信号肽内。

    判定依据（真核与原核信号肽的公认结构特征）
    ------------------------------------------
    信号肽由三段构成：带正电的 n-region、**疏水核心 h-region（7-15 个疏水残基）**、
    以及含小残基的 c-region（切割位点）。其中 h-region 最具辨识度，因此：

    1. 在前 ``max_scan`` 个残基内寻找最长的连续疏水段；
    2. 长度 ≥ 7 且起始位置 ≤ 30 才认定为 h-region；
    3. 信号肽终点取 h-region 之后约 6 个残基。

    **这是启发式，不是 SignalP**。区间会在扫描计划中被显式记录（原因写明"启发式识别"），
    用户可通过配置关闭。若要精确判定，建议接入 SignalP 并替换本函数。
    """
    limit = min(max_scan, len(sequence))
    best_start = -1
    best_length = 0

    index = 0
    while index < limit:
        if sequence[index] in SIGNAL_HYDROPHOBIC:
            run_start = index
            while index < limit and sequence[index] in SIGNAL_HYDROPHOBIC:
                index += 1
            run_length = index - run_start
            if run_length > best_length:
                best_length = run_length
                best_start = run_start
        else:
            index += 1

    if best_length < 7 or best_start < 0 or best_start > 30:
        return None

    end = min(len(sequence), best_start + best_length + 6)
    # 区间过短或几乎覆盖全长（可能是整体疏水的膜蛋白）时不作保护
    if end < 12 or end > len(sequence) * 0.5:
        return None
    return (0, end)


@dataclass
class ChargeProfile:
    """净电荷随 pH 变化曲线。"""

    ph_values: list[float]
    charges: list[float]

    @property
    def isoelectric_point(self) -> float:
        """曲线过零点（插值）。"""
        for index in range(len(self.charges) - 1):
            high, low = self.charges[index], self.charges[index + 1]
            if high >= 0 >= low:
                span = high - low
                if span == 0:
                    return self.ph_values[index]
                ratio = high / span
                return round(
                    self.ph_values[index]
                    + ratio * (self.ph_values[index + 1] - self.ph_values[index]),
                    2,
                )
        return 7.0


def charge_profile(sequence: str, ph_min: float = 0.0, ph_max: float = 14.0, step: float = 0.25) -> ChargeProfile:
    """生成净电荷-pH 曲线。"""
    ph_values: list[float] = []
    charges: list[float] = []
    steps = int(round((ph_max - ph_min) / step)) + 1
    for index in range(steps):
        ph = round(ph_min + index * step, 3)
        ph_values.append(ph)
        charges.append(net_charge(sequence, ph))
    return ChargeProfile(ph_values=ph_values, charges=charges)
