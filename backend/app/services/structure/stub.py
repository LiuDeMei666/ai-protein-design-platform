"""离线占位 Provider。

用途
----
1. **断网兜底**：企业内网离线且未部署本地 ESMFold 时，平台仍能跑通全流程，
   前端界面、图表与作业队列不至于整体不可用。
2. **单元测试**：提供确定性结构，使测试不依赖网络与 GPU。

刻意设计
--------
* 生成理想 α-螺旋骨架几何（真实可解析、DSSP 可识别），但 **pLDDT 固定压低到
  30-45**，让前端呈现"极低置信度"配色，用户一眼就能看出这是占位结构。
* ``source`` / ``degradation_reason`` 明确标注 "stub"，任何导出报告都会带上。
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from ...core.config import get_settings
from ...core.logging import get_logger
from . import pdb_utils
from .base import BasePredictor, StructureResult

logger = get_logger(__name__)

# 理想 α-螺旋几何参数
HELIX_RADIUS = 2.30  # CA 到螺旋轴的距离（Å）
HELIX_RISE = 1.50  # 每残基沿轴上升（Å）
HELIX_TURN = math.radians(100.0)  # 每残基旋转角

# 相对 CA 的骨架原子偏移（近似理想几何，单位 Å）
# 注意：CA 必须在列，因为 pLDDT 存放于 CA 的 B-factor 列，缺了它整个置信度链条就断了
BACKBONE_OFFSETS: dict[str, tuple[float, float, float]] = {
    "N": (-0.55, 1.05, 0.40),
    "CA": (0.00, 0.00, 0.00),
    "C": (0.60, 0.55, -0.55),
    "O": (1.05, 0.35, -1.15),
    "CB": (-0.75, -0.95, -0.75),
}

#: PDB 第 18-20 列必须是**三字母**残基码；写入 1 字母码会导致解析器查表失败
ONE_TO_THREE: dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _format_atom_line(
    serial: int,
    name: str,
    resname: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    bfactor: float,
    chain: str = "A",
    occupancy: float = 1.0,
) -> str:
    """按 PDB 规范的**固定列宽**生成 ATOM 行。

    列位置必须严格对齐，否则解析器（含 3Dmol.js）会读错字段：

    ======  ==========================
    列      内容
    ======  ==========================
    1-6     ``ATOM  ``
    7-11    原子序号
    13-16   原子名（单字符元素按惯例从第 14 列起）
    18-20   残基三字母码
    22      链标识
    23-26   残基序号
    31-54   x/y/z
    55-60   占据率
    61-66   B-factor（此处存 pLDDT）
    77-78   元素符号
    ======  ==========================
    """
    element = name[0]
    name_field = f" {name:<3s}" if len(name) < 4 else name
    return (
        f"ATOM  {serial:5d} {name_field} {resname:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{bfactor:6.2f}          {element:>2s}"
    )


class StubPredictor(BasePredictor):
    """确定性占位结构生成器。"""

    name = "stub"
    model_version = "stub-helix-v1"
    max_length: int | None = None  # 无长度限制（本地生成，成本恒定）

    def available(self) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": True,
            "max_length": None,
            "model_version": self.model_version,
            "note": "离线占位结构（理想 α-螺旋骨架），置信度被刻意压低，仅供界面演示与测试",
        }

    @staticmethod
    def _seed(sequence: str) -> int:
        return int(hashlib.sha256(sequence.encode()).hexdigest()[:8], 16)

    def _plddt(self, sequence: str, index: int) -> float:
        """确定性伪随机 pLDDT，落在 30-45（明显不可信区间）。"""
        seed = self._seed(sequence)
        mixed = (seed ^ (index * 2654435761)) & 0xFFFFFFFF
        return 30.0 + (mixed % 1500) / 100.0

    def predict(self, sequence: str, *, allow_split: bool = True) -> StructureResult:
        sequence = self._require_sequence(sequence)
        length = len(sequence)
        logger.warning("使用 stub Provider 生成占位结构 %s", length)

        lines: list[str] = [
            "HEADER    STUB PLACEHOLDER STRUCTURE (NOT A REAL PREDICTION)",
            "REMARK   1 本结构由离线占位 Provider 生成，仅用于界面演示与自动化测试。",
            "REMARK   1 几何为理想 alpha-螺旋骨架，不具备真实的序列-结构对应关系。",
        ]

        serial = 1
        atom_lines: list[str] = []
        for index, residue in enumerate(sequence):
            angle = index * HELIX_TURN
            ca_x = HELIX_RADIUS * math.cos(angle)
            ca_y = HELIX_RADIUS * math.sin(angle)
            ca_z = index * HELIX_RISE
            plddt = self._plddt(sequence, index)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            resname3 = ONE_TO_THREE.get(residue, "ALA")

            for atom_name, offset in BACKBONE_OFFSETS.items():
                if atom_name == "CB" and residue == "G":
                    continue  # 甘氨酸无 CB
                # 把偏移随螺旋一起旋转，保持局部几何合理
                ox, oy, oz = offset
                x = ca_x + ox * cos_a - oy * sin_a
                y = ca_y + ox * sin_a + oy * cos_a
                z = ca_z + oz
                atom_lines.append(
                    _format_atom_line(serial, atom_name, resname3, index + 1, x, y, z, plddt)
                )
                serial += 1
            atom_lines.append(f"TER   {serial:5d}      {resname3:>3s} A{index + 1:4d}")
            serial += 1

        lines.extend(atom_lines)
        lines.append("END")
        pdb_text = "\n".join(lines) + "\n"

        data = pdb_utils.parse_pdb(pdb_text, source=self.name)
        plddt_values = pdb_utils.extract_plddt(data)
        stats = pdb_utils.build_structure_stats(data, plddt_values)
        stats["stub"] = True

        return StructureResult(
            pdb_text=pdb_text,
            plddt=plddt_values,
            mean_plddt=round(float(sum(plddt_values) / len(plddt_values)), 2) if plddt_values else 0.0,
            source=self.name,
            truncated=False,
            segments=[(0, length)],
            model_version=self.model_version,
            degradation_reason=(
                "未使用真实结构预测：当前使用离线占位 Provider，"
                "结构不具备生物学意义，仅用于流程演示与测试。"
            ),
            stats=stats,
            warnings=["此为占位结构，请勿用于任何实验决策。"],
        )
