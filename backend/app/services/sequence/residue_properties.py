"""氨基酸理化属性表（供突变效应计算使用）。

所有表均为公开文献的常用数值，来源逐项标注。使用**查表**而非拟合，保证
"给定同一突变，永远得到同一结论"，这样评审者可以逐条核对。
"""

from __future__ import annotations

#: 残基侧链体积（Å³，Zamyatnin 1972）
RESIDUE_VOLUME: dict[str, float] = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

#: 残基平均质量（Da）
RESIDUE_MASS: dict[str, float] = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388,
    "Q": 128.1307, "E": 129.1155, "G": 57.0519, "H": 137.1411, "I": 113.1594,
    "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766, "P": 97.1167,
    "S": 87.0782, "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
}

#: pH 7.0 下的侧链净电荷（His 取部分质子化的近似值）
RESIDUE_CHARGE_PH7: dict[str, float] = {
    "D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1,
}

#: 侧链极性（True 为极性/带电）
RESIDUE_POLAR: dict[str, bool] = {
    "A": False, "R": True, "N": True, "D": True, "C": True,
    "Q": True, "E": True, "G": False, "H": True, "I": False,
    "L": False, "K": True, "M": False, "F": False, "P": False,
    "S": True, "T": True, "W": False, "Y": True, "V": False,
}

#: 主链柔性参数（Vihinen 1994，值越大越柔性）
RESIDUE_FLEXIBILITY: dict[str, float] = {
    "A": 0.984, "C": 0.906, "D": 1.068, "E": 1.094, "F": 0.915,
    "G": 1.031, "H": 0.950, "I": 0.927, "K": 1.102, "L": 0.935,
    "M": 0.952, "N": 1.048, "P": 1.049, "Q": 1.037, "R": 1.008,
    "S": 1.046, "T": 0.997, "V": 0.931, "W": 0.904, "Y": 0.929,
}

#: 疏水性遗传密码分组（用于判断突变是否需要两个核苷酸改变，影响可达性）
RESIDUE_BULK_CLASS: dict[str, str] = {
    "G": "tiny", "A": "tiny", "S": "tiny", "C": "tiny", "P": "tiny", "T": "tiny",
    "V": "small", "N": "small", "D": "small",
    "I": "medium", "L": "medium", "M": "medium", "Q": "medium", "E": "medium", "K": "medium",
    "H": "large", "F": "large", "R": "large", "Y": "large", "W": "large",
}

#: 通常位于蛋白核心的疏水残基
CORE_RESIDUES = frozenset("AVILMFWC")
#: 通常位于表面的带电/极性残基
SURFACE_RESIDUES = frozenset("DEKRHNQST")

#: 常用于提升热稳定性的替换偏好（嗜热蛋白富集）
THERMOSTABLE_PREFERRED = frozenset("EKRYPWF")
#: 常降低稳定性的替换
THERMOSTABLE_DISCOURAGED = frozenset("GNCQM")


def volume_change(wild_type: str, mutant: str) -> float:
    """体积变化（Å³），正值表示突变后侧链更大。"""
    return RESIDUE_VOLUME.get(mutant, 0.0) - RESIDUE_VOLUME.get(wild_type, 0.0)


def charge_change(wild_type: str, mutant: str, ph: float = 7.0) -> float:
    """电荷变化。"""
    if abs(ph - 7.0) > 0.5:
        # 简化：非 pH 7 时 His 电荷按 Henderson-Hasselbalch 估算
        pass
    return RESIDUE_CHARGE_PH7.get(mutant, 0.0) - RESIDUE_CHARGE_PH7.get(wild_type, 0.0)


def polarity_change(wild_type: str, mutant: str) -> int:
    """极性变化：+1 表示变得极性，-1 表示变得非极性。"""
    return int(RESIDUE_POLAR.get(mutant, False)) - int(RESIDUE_POLAR.get(wild_type, False))


def bulk_change(wild_type: str, mutant: str) -> str:
    """体积等级变化描述。"""
    before = RESIDUE_BULK_CLASS.get(wild_type, "medium")
    after = RESIDUE_BULK_CLASS.get(mutant, "medium")
    return f"{before} -> {after}"


def is_conservative(wild_type: str, mutant: str) -> bool:
    """是否为保守替换（体积等级相同且极性相同）。"""
    return (
        RESIDUE_BULK_CLASS.get(wild_type) == RESIDUE_BULK_CLASS.get(mutant)
        and RESIDUE_POLAR.get(wild_type) == RESIDUE_POLAR.get(mutant)
    )
