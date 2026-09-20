"""PDB 解析与结构分析。

不依赖 Biopython 的结构对象，直接用 numpy + scipy 实现，原因有三：
1. ESMFold 产出的 PDB 结构简单（单链、无 HETATM、无 MODEL），自研解析更快；
2. 需要把 pLDDT（B-factor 列）与残基严格对齐，自研更可控；
3. 便于在无 Biopython 的极简部署中降级运行。

实现的算法
----------
* **pLDDT 提取**：ESMFold 把逐残基置信度写入 CA 原子的 B-factor。
* **SASA**：Shrake-Rupley 数值积分（92 点球面采样），再用 Tien 2013 理论最大值
  换算相对暴露度，用于判定疏水核心暴露。
* **二级结构**：DSSP-lite —— 按 Kabsch-Sander 氢键能量判据识别 i→i+3/4/5 转角
  与 β 桥，输出 H/G/I/E/- 五态。这是近似算法，报告中会标注"计算近似"。
* **接触图**：CA-CA 距离阈值法。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...core.logging import get_logger

logger = get_logger(__name__)

# 范德华半径（Å），用于 SASA
ELEMENT_RADII: dict[str, float] = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "H": 1.20}

# Tien et al. (2013) 理论最大可及表面积（Å²），用于相对 SASA
MAX_ASA: dict[str, float] = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "Q": 225.0, "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}

# 3 字母 -> 1 字母
THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}

BACKBONE_ATOMS = ("N", "CA", "C", "O")


@dataclass
class Atom:
    """单个原子。"""

    name: str
    element: str
    coord: np.ndarray
    bfactor: float
    occupancy: float = 1.0


@dataclass
class Residue:
    """单个残基。"""

    index: int  # 0-based 序列序号
    number: int  # PDB 残基号
    name3: str
    name1: str
    chain: str
    atoms: dict[str, Atom] = field(default_factory=dict)

    @property
    def ca(self) -> np.ndarray | None:
        atom = self.atoms.get("CA")
        return atom.coord if atom else None

    @property
    def is_amino_acid(self) -> bool:
        return self.name1 in MAX_ASA or self.name1 in {"U", "O"}


@dataclass
class StructureData:
    """解析后的结构。"""

    residues: list[Residue]
    source: str = ""

    @property
    def length(self) -> int:
        return len(self.residues)

    @property
    def sequence(self) -> str:
        return "".join(residue.name1 for residue in self.residues)

    def ca_coords(self) -> np.ndarray:
        coords = [residue.ca for residue in self.residues if residue.ca is not None]
        if not coords:
            return np.zeros((0, 3), dtype=np.float64)
        return np.vstack(coords)

    def atom_cloud(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (坐标 [N,3], 半径 [N], 所属残基下标 [N])，用于 SASA。"""
        coords: list[np.ndarray] = []
        radii: list[float] = []
        owners: list[int] = []
        for residue in self.residues:
            for atom in residue.atoms.values():
                if atom.element == "H":
                    continue
                radius = ELEMENT_RADII.get(atom.element)
                if radius is None:
                    continue
                coords.append(atom.coord)
                radii.append(radius)
                owners.append(residue.index)
        if not coords:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.int64),
            )
        return (
            np.asarray(coords, dtype=np.float64),
            np.asarray(radii, dtype=np.float64),
            np.asarray(owners, dtype=np.int64),
        )


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def parse_pdb(pdb_text: str, source: str = "") -> StructureData:
    """解析 PDB 文本。

    多模型文件（存在 ``MODEL``）只取第一个模型；按 ``(chain, resnum, icode)``
    分组，保持文件中出现顺序作为 0-based 残基序号。
    """
    residues: list[Residue] = []
    index_by_key: dict[tuple[str, int, str], int] = {}
    seen_model = False
    finished_first_model = False

    for raw_line in pdb_text.splitlines():
        record = raw_line[:6].strip()
        if record == "MODEL":
            if seen_model:
                finished_first_model = True
            seen_model = True
            continue
        if record == "ENDMDL" and seen_model:
            break
        if finished_first_model:
            break
        if record not in ("ATOM", "HETATM"):
            continue

        try:
            name = raw_line[12:16].strip()
            resname = raw_line[17:20].strip().upper()
            chain = raw_line[21:22].strip() or "A"
            number = int(raw_line[22:26])
            icode = raw_line[26:27].strip()
            x = float(raw_line[30:38])
            y = float(raw_line[38:46])
            z = float(raw_line[46:54])
            bfactor = float(raw_line[60:66]) if raw_line[60:66].strip() else 0.0
            occupancy = float(raw_line[54:60]) if raw_line[54:60].strip() else 1.0
        except (ValueError, IndexError):
            continue

        name1 = THREE_TO_ONE.get(resname)
        if name1 is None:
            # 跳过水、离子与配体；对结构分析无贡献
            continue

        key = (chain, number, icode)
        if key not in index_by_key:
            index_by_key[key] = len(residues)
            residues.append(
                Residue(
                    index=len(residues),
                    number=number,
                    name3=resname,
                    name1=name1,
                    chain=chain,
                )
            )
        residue = residues[index_by_key[key]]
        element = (raw_line[76:78].strip() or name[:1]).upper()
        residue.atoms[name] = Atom(
            name=name,
            element=element,
            coord=np.array([x, y, z], dtype=np.float64),
            bfactor=bfactor,
            occupancy=occupancy,
        )

    return StructureData(residues=residues, source=source)


def raw_plddt(data: StructureData) -> list[float]:
    """从 CA 原子的 B-factor 取原始值（未做量纲归一化）。"""
    values: list[float] = []
    for residue in data.residues:
        atom = residue.atoms.get("CA")
        values.append(float(atom.bfactor) if atom else 0.0)
    return values


def plddt_scale_factor(raw_values: list[float]) -> float:
    """推断 pLDDT 量纲。

    **实测差异**：ESM Atlas 在线接口返回的 B-factor 是 0-1 的分数
    （如 0.54~0.96），而本地 ESMFold 与 AlphaFold 输出 0-100。这里用
    "最大值 <= 1.5 即判定为分数制" 自动归一化，避免把 0.93 当成 0.93 分的
    荒谬置信度。判定结果会写入 ``stats.plddt_scale`` 供人工核对。
    """
    if not raw_values:
        return 1.0
    return 100.0 if max(raw_values) <= 1.5 else 1.0


def extract_plddt(data: StructureData, scale_to_100: bool = True) -> list[float]:
    """从 CA 原子的 B-factor 提取逐残基 pLDDT（默认归一到 0-100）。"""
    values = raw_plddt(data)
    if scale_to_100:
        factor = plddt_scale_factor(values)
        values = [value * factor for value in values]
    return [round(value, 2) for value in values]


# --------------------------------------------------------------------------- #
# 几何
# --------------------------------------------------------------------------- #
def radius_of_gyration(data: StructureData) -> float:
    """回旋半径（Å），衡量结构紧凑程度。"""
    coords = data.ca_coords()
    if len(coords) < 2:
        return 0.0
    centroid = coords.mean(axis=0)
    return float(np.sqrt(((coords - centroid) ** 2).sum(axis=1).mean()))


def _sphere_points(n: int = 92) -> np.ndarray:
    """黄金螺旋法生成球面均匀采样点。"""
    indices = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = math.pi * (1 + 5**0.5) * indices
    points = np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    )
    return points


_SPHERE_92 = _sphere_points(92)


def shrake_rupley_sasa(
    data: StructureData, probe_radius: float = 1.4, n_points: int = 92
) -> dict[int, float]:
    """Shrake-Rupley 算法计算每个残基的可及表面积（Å²）。

    实现要点：用 ``cKDTree`` 做邻居查询，把每个原子的采样点中"未被邻近原子覆盖"
    的比例乘以球面积，得到该原子暴露面积，再按残基汇总。
    """
    from scipy.spatial import cKDTree

    coords, radii, owners = data.atom_cloud()
    if len(coords) == 0:
        return {}

    expanded = radii + probe_radius
    points = _SPHERE_92[:n_points]
    tree = cKDTree(coords)
    # 单原子最大扩展半径 1.8 + 1.4 = 3.2，取 6.4 作为邻居搜索半径足够
    neighbor_lists = tree.query_ball_point(coords, r=6.4)

    atom_sasa = np.zeros(len(coords), dtype=np.float64)
    for atom_index, (center, radius) in enumerate(zip(coords, expanded)):
        neighbours = neighbor_lists[atom_index]
        if len(neighbours) <= 1:
            atom_sasa[atom_index] = 4 * math.pi * radius**2
            continue

        neighbour_coords = coords[neighbours]
        neighbour_radii = expanded[neighbours]
        sample_points = center + points * radius  # [P, 3]
        # 到各邻居中心的距离
        deltas = sample_points[:, None, :] - neighbour_coords[None, :, :]
        distances = np.sqrt((deltas**2).sum(axis=2))  # [P, K]
        covered = (distances < neighbour_radii[None, :]).any(axis=1)
        exposed_ratio = 1.0 - covered.mean()
        atom_sasa[atom_index] = exposed_ratio * 4 * math.pi * radius**2

    residue_sasa: dict[int, float] = {}
    for owner, value in zip(owners, atom_sasa):
        residue_sasa[int(owner)] = residue_sasa.get(int(owner), 0.0) + float(value)
    return {key: round(value, 3) for key, value in residue_sasa.items()}


def relative_sasa(data: StructureData, sasa: dict[int, float] | None = None) -> dict[int, float]:
    """相对暴露度 = 实测 SASA / 理论最大 SASA（0-1+）。"""
    sasa = sasa if sasa is not None else shrake_rupley_sasa(data)
    result: dict[int, float] = {}
    for residue in data.residues:
        maximum = MAX_ASA.get(residue.name1)
        if not maximum:
            continue
        result[residue.index] = round(sasa.get(residue.index, 0.0) / maximum, 4)
    return result


# --------------------------------------------------------------------------- #
# 二级结构（DSSP-lite）
# --------------------------------------------------------------------------- #
_HBOND_ENERGY_COEFF = 0.084 * 332.0  # kcal/mol


def _virtual_hydrogen(residues: list[Residue], index: int) -> np.ndarray | None:
    """按 DSSP 约定估算酰胺氢位置。"""
    if index == 0:
        return None
    current = residues[index].atoms.get("N")
    previous_c = residues[index - 1].atoms.get("C")
    if current is None or previous_c is None:
        return None
    direction = current.coord - previous_c.coord
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None
    return current.coord + direction / norm * 1.0


def _hbond_energy(donor_n: np.ndarray, hydrogen: np.ndarray, acceptor_c: np.ndarray, acceptor_o: np.ndarray) -> float:
    """Kabsch-Sander 氢键能量（kcal/mol），低于 -0.5 判为氢键。"""
    r_on = np.linalg.norm(acceptor_o - donor_n)
    r_ch = np.linalg.norm(acceptor_c - hydrogen)
    r_oh = np.linalg.norm(acceptor_o - hydrogen)
    r_cn = np.linalg.norm(acceptor_c - donor_n)
    if min(r_on, r_ch, r_oh, r_cn) < 1e-3:
        return 0.0
    return _HBOND_ENERGY_COEFF * (1 / r_on + 1 / r_ch - 1 / r_oh - 1 / r_cn)


def dssp_lite(data: StructureData) -> str:
    """简化的二级结构指派，返回长度等于残基数的字符串。

    状态字符：``H`` α-螺旋、``G`` 3-10 螺旋、``I`` π-螺旋、``E`` β-折叠、``-`` 无规卷曲。

    方向约定（**易错点**）
    ---------------------
    DSSP 的 n-turn 定义是"**CO(i)···HN(i+n)** 的氢键"，即 **给体在 i+n、受体在 i**。
    因此 α-螺旋对应 ``hb(i, i+4)``（CO(i) 与 HN(i+4) 成键），而不是反向。
    早期实现把方向写反会导致整条序列被判为无规卷曲。
    """
    residues = data.residues
    length = len(residues)
    if length < 4:
        return "-" * length

    # 预计算氢键：记录 (给体残基, 受体残基)，即 HN(donor)···CO(acceptor)
    hbonds: set[tuple[int, int]] = set()
    backbone: list[dict[str, np.ndarray]] = []
    for residue in residues:
        entry: dict[str, np.ndarray] = {}
        for name in BACKBONE_ATOMS:
            atom = residue.atoms.get(name)
            if atom is not None:
                entry[name] = atom.coord
        backbone.append(entry)

    hydrogens = [_virtual_hydrogen(residues, i) for i in range(length)]

    for acceptor in range(length):
        acceptor_c = backbone[acceptor].get("C")
        acceptor_o = backbone[acceptor].get("O")
        if acceptor_c is None or acceptor_o is None:
            continue
        for donor in range(length):
            if abs(donor - acceptor) < 2:
                continue
            hydrogen = hydrogens[donor]
            donor_n = backbone[donor].get("N")
            if hydrogen is None or donor_n is None:
                continue
            if _hbond_energy(donor_n, hydrogen, acceptor_c, acceptor_o) < -0.5:
                hbonds.add((donor, acceptor))

    def has_hbond(donor: int, acceptor: int) -> bool:
        return (donor, acceptor) in hbonds

    def hb(acceptor: int, donor: int) -> bool:
        """DSSP 记法 Hbond(a, b) = CO(a)···HN(b)。"""
        return has_hbond(donor, acceptor)

    def n_turn(start: int, n: int) -> bool:
        """n-turn(start)：CO(start)···HN(start+n)。"""
        return 0 <= start and start + n < length and hb(start, start + n)

    structure = ["-"] * length

    # --- 螺旋：DSSP 规则为"4-turn(i) 与 4-turn(i-1) 同时成立 -> 残基 i..i+3 为 H" ---
    for i in range(1, length):
        if n_turn(i, 4) and n_turn(i - 1, 4):
            for k in range(i, min(i + 4, length)):
                structure[k] = "H"
    for i in range(1, length):
        if n_turn(i, 3) and n_turn(i - 1, 3):
            for k in range(i, min(i + 3, length)):
                if structure[k] == "-":
                    structure[k] = "G"
    for i in range(1, length):
        if n_turn(i, 5) and n_turn(i - 1, 5):
            for k in range(i, min(i + 5, length)):
                if structure[k] == "-":
                    structure[k] = "I"

    # --- β-桥：DSSP 的平行 / 反平行判据 ---
    for i in range(1, length - 1):
        for j in range(i + 3, length - 1):
            parallel = (hb(i - 1, j) and hb(j, i + 1)) or (hb(j - 1, i) and hb(i, j + 1))
            antiparallel = (hb(i, j) and hb(j, i)) or (hb(i - 1, j + 1) and hb(j - 1, i + 1))
            if parallel or antiparallel:
                for k in (i, j):
                    if structure[k] in ("-", "G", "I"):
                        structure[k] = "E"

    return "".join(structure)


def secondary_structure_composition(structure: str) -> dict[str, float]:
    """二级结构组成占比。"""
    total = len(structure) or 1
    counts = {"helix": 0, "sheet": 0, "coil": 0, "helix_310": 0, "helix_pi": 0}
    for char in structure:
        if char == "H":
            counts["helix"] += 1
        elif char == "G":
            counts["helix_310"] += 1
        elif char == "I":
            counts["helix_pi"] += 1
        elif char == "E":
            counts["sheet"] += 1
        else:
            counts["coil"] += 1
    composition = {key: round(value / total, 4) for key, value in counts.items()}
    composition["helix_total"] = round(
        composition["helix"] + composition["helix_310"] + composition["helix_pi"], 4
    )
    return composition


# --------------------------------------------------------------------------- #
# 接触图与疏水暴露
# --------------------------------------------------------------------------- #
def contact_map(data: StructureData, max_length: int = 600, threshold: float = 8.0) -> list[list[int]]:
    """CA-CA 接触对（距离 < ``threshold`` Å），返回 ``[[i, j], ...]``。

    超过 ``max_length`` 时返回空列表：前端渲染成本与信息增益不成比例。
    """
    if data.length > max_length:
        return []
    coords = data.ca_coords()
    if len(coords) < 2:
        return []
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=threshold, output_type="ndarray")
    if len(pairs) == 0:
        return []
    # 只保留 |i-j| >= 2 的接触，并限制数量
    mask = np.abs(pairs[:, 0] - pairs[:, 1]) >= 2
    pairs = pairs[mask]
    if len(pairs) > 20000:
        step = max(1, len(pairs) // 20000)
        pairs = pairs[::step]
    return pairs.tolist()


def hydrophobic_exposure(
    data: StructureData,
    sasa: dict[int, float] | None = None,
    cutoff: float = 0.25,
) -> dict[str, object]:
    """疏水核心暴露分析。

    核心残基（疏水性高）若相对 SASA 偏高，说明该处折叠松散、潜藏聚集风险。
    """
    relative = relative_sasa(data, sasa)
    hydrophobic = set("AVILMFWC")
    buried: list[int] = []
    exposed: list[int] = []
    for residue in data.residues:
        if residue.name1 not in hydrophobic:
            continue
        value = relative.get(residue.index)
        if value is None:
            continue
        if value > cutoff:
            exposed.append(residue.index)
        else:
            buried.append(residue.index)

    total = len(buried) + len(exposed)
    return {
        "buried_count": len(buried),
        "exposed_count": len(exposed),
        "exposed_ratio": round(len(exposed) / total, 4) if total else 0.0,
        "exposed_positions": exposed[:200],
        "algorithm": "Shrake-Rupley SASA + Tien 2013 理论最大值",
    }


def mean_relative_sasa(data: StructureData, sasa: dict[int, float] | None = None) -> float:
    """平均相对暴露度，用于表达量/溶解性的结构特征。"""
    relative = relative_sasa(data, sasa)
    if not relative:
        return 0.0
    return round(float(np.mean(list(relative.values()))), 4)


def flexible_region_ratio(data: StructureData, plddt: list[float] | None = None) -> float:
    """柔性区域占比：pLDDT < 70 的残基比例（ESMFold 语境下近似无序区）。"""
    values = plddt if plddt is not None else extract_plddt(data)
    if not values:
        return 0.0
    return round(sum(1 for value in values if value < 70) / len(values), 4)


def build_structure_stats(data: StructureData, plddt: list[float] | None = None) -> dict[str, object]:
    """汇总结构统计信息，写入 :class:`StructureResult.stats`。"""
    from ...core.config import load_platform_config

    config = load_platform_config().get("structure", {})
    bands_config = config.get("plddt_bands", {})
    very_high = float(bands_config.get("very_high", 90))
    confident = float(bands_config.get("confident", 70))
    low = float(bands_config.get("low", 50))
    sasa_cutoff = float(config.get("burial_sasa_cutoff", 0.25))
    map_max = int(config.get("contact_map_max_length", 600))

    values = plddt if plddt is not None else extract_plddt(data)
    secondary = dssp_lite(data)
    sasa = shrake_rupley_sasa(data)
    relative = relative_sasa(data, sasa)

    total = len(values) or 1
    bands = {"very_high": 0, "confident": 0, "low": 0, "very_low": 0}
    for value in values:
        if value >= very_high:
            bands["very_high"] += 1
        elif value >= confident:
            bands["confident"] += 1
        elif value >= low:
            bands["low"] += 1
        else:
            bands["very_low"] += 1

    return {
        "plddt_scale": plddt_scale_factor(raw_plddt(data)),
        "mean_plddt": round(float(sum(values) / total), 2) if values else 0.0,
        "plddt_bands": {key: round(count / total, 4) for key, count in bands.items()},
        "secondary_structure": secondary,
        "secondary_structure_composition": secondary_structure_composition(secondary),
        "radius_of_gyration": round(radius_of_gyration(data), 3),
        "mean_relative_sasa": mean_relative_sasa(data, sasa),
        # 逐残基数组必须输出：突变设计需要"位点级"的结构先验
        # （pLDDT = 该位点是否刚性；relative_sasa = 该位点是否埋藏）
        "plddt": [round(float(value), 2) for value in values],
        "relative_sasa": [relative.get(index, 0.0) for index in range(len(data.residues))],
        "hydrophobic_exposure": hydrophobic_exposure(data, sasa, sasa_cutoff),
        "flexible_region_ratio": flexible_region_ratio(data, values),
        "contacts": contact_map(data, map_max),
        "algorithms": {
            "plddt": "ESMFold B-factor",
            "sasa": "Shrake-Rupley (92 points, probe 1.4Å)",
            "secondary_structure": "DSSP-lite (Kabsch-Sander 氢键能量)",
            "contact_map": "CA-CA < 8Å",
        },
    }


def residue_plddt_map(data: StructureData) -> list[dict[str, object]]:
    """逐残基摘要，供前端热点联动。"""
    return [
        {
            "index": residue.index,
            "number": residue.number,
            "residue": residue.name1,
            "plddt": round(float(residue.atoms["CA"].bfactor), 2) if "CA" in residue.atoms else 0.0,
        }
        for residue in data.residues
    ]
