"""人工指定突变的规划与导出（MVP 验证工作流）。

与自动模式的根本区别
--------------------
自动模式（``single`` / ``combination`` / ``local``）由平台扫描
**全部可突变位点 × 19 种氨基酸**，再从结果里排序推荐；人工模式反过来：
**位点、替换氨基酸、锁定位点全部由人给出**，平台只负责按既有评分链路逐条评估。
对应需求"验证 MVP 阶段开发"：::

    人工选表位位点 → 人工写单点突变 FASTA（20~30 条，含野生对照）
    → 批量提交 → 导出对比野生型 → 校验打分趋势

为什么不做"自动识别关键位点"
----------------------------
需求明确要求锁定位点（如二硫键 Cys）**完全由人工指定**。平台不去猜测
"哪些残基互相形成二硫键"，因为结构模型给出的 Cys 配对本身带不确定性，
一旦平台自作主张锁错位点，人工选定的表位突变反而会被静默剔除。
本模块只做**约束校验**：人工声明的锁定位点绝不允许被突变，冲突时明确报错
并逐条列出原因，绝不静默丢弃（这是全平台的不变量）。

本模块**不引入任何新的预测能力**：规划阶段是纯字符串与集合运算，
不调用 ESM-2、不查结构、不落库，因此可以同步返回，供前端做提交前预览。
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any, Iterable

from ...core.config import load_platform_config
from ...core.errors import ValidationError
from .rulepacks import get_rulepack
from .scanner import AA_ORDER

# --------------------------------------------------------------------------- #
# 默认候选氨基酸
# --------------------------------------------------------------------------- #

#: 残基按侧链性质归类（仅用于给出**默认**候选，人工可任意覆盖）
CHARGE_POSITIVE = frozenset("KRH")
CHARGE_NEGATIVE = frozenset("DE")
AROMATIC = frozenset("FWY")
HYDROPHOBIC = frozenset("LIVMA")
POLAR = frozenset("STNQ")
BACKBONE_SPECIAL = frozenset("GPC")

#: 每位点默认给出的候选氨基酸。
#:
#: 设计依据来自需求原文："优先选不带电荷 / 改变电荷、降低表面暴露疏水的氨基酸"，
#: 再叠加丙氨酸扫描（Ala-scanning）这一标准做法——每个位点都包含 ``A``。
#: 每类固定 3~4 个，保证**结果可复现**且单批规模落在需求建议的 20~30 条内
#: （5 个位点 × 4 = 20 条突变 + 1 条野生对照）。
DEFAULT_SUBSTITUTIONS: dict[str, tuple[str, ...]] = {
    # 正电 -> 消除电荷（A/N/Q）+ 反转电荷（E）
    "positive": ("A", "N", "Q", "E"),
    # 负电 -> 消除电荷（A/S/T）+ 反转电荷（K）
    "negative": ("A", "S", "T", "K"),
    # 芳香 -> 打掉芳环并降低表面疏水
    "aromatic": ("A", "S", "L", "T"),
    # 疏水 -> 引入极性、降低表面暴露疏水
    "hydrophobic": ("A", "S", "T", "N"),
    # 极性不带电 -> 引入电荷 / 增加疏水，探测两个方向
    "polar": ("A", "D", "K", "L"),
    # Gly/Pro/Cys 骨架特殊（Gly 提供柔性、Pro 固定主链、Cys 可能成二硫键），
    # 只做温和替换，不主动引入新的骨架扰动
    "special": ("A", "S", "T"),
}


#: 单批人工突变的数量上限。
#:
#: 需求"风险控制要点"第 1 条明确要求"不要一次性造上百条突变，20~30 条足够验证 MVP"。
#: 这里设一个远高于建议值的硬上限，目的不是限制正常使用，而是防止误把
#: 位点或替换列表粘错（例如整列粘贴）导致一次提交几千条、把 ESM-2 打分
#: 与数据库写入打满。超限时**明确报错并给出拆分做法**，不做静默截断。
MAX_MUTATIONS = 500

#: 需求建议的单批规模（仅用于提示，不做强制）
RECOMMENDED_BATCH = (20, 30)


def residue_class(wild_type: str) -> str:
    """把残基归入默认候选取值的类别。"""
    if wild_type in BACKBONE_SPECIAL:
        return "special"
    if wild_type in CHARGE_POSITIVE:
        return "positive"
    if wild_type in CHARGE_NEGATIVE:
        return "negative"
    if wild_type in AROMATIC:
        return "aromatic"
    if wild_type in HYDROPHOBIC:
        return "hydrophobic"
    if wild_type in POLAR:
        return "polar"
    return "special"


def default_substitutions(wild_type: str) -> tuple[str, ...]:
    """该残基的默认候选氨基酸（不含野生型自身）。"""
    candidates = DEFAULT_SUBSTITUTIONS[residue_class(wild_type)]
    return tuple(item for item in candidates if item != wild_type)


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BlockedTarget:
    """一个被拒绝的目标位点及其原因。"""

    position: int  # 1-based，与前端一致
    residue: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"position": self.position, "residue": self.residue, "reason": self.reason}


@dataclass(frozen=True)
class ManualMutation:
    """一条人工指定的单点突变。"""

    position: int  # 0-based，内部计算用
    wild_type: str
    mutant: str
    label: str  # W23A
    sequence: str  # 突变后的完整序列
    sequence_id: str  # 供湿实验合成与登记使用的唯一编号
    note: str = ""

    @property
    def display_position(self) -> int:
        return self.position + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position + 1,
            "wild_type": self.wild_type,
            "mutant": self.mutant,
            "label": self.label,
            "sequence": self.sequence,
            "sequence_id": self.sequence_id,
            "sequence_length": len(self.sequence),
            "note": self.note,
        }

    def fasta_record(self, width: int = 60) -> str:
        """标准 FASTA 记录文本。"""
        lines = [f">{self.sequence_id}"]
        for start in range(0, len(self.sequence), width):
            lines.append(self.sequence[start : start + width])
        return "\n".join(lines)


@dataclass
class ManualPlan:
    """人工突变清单。

    ``blocked`` 非空即表示**有人工输入被拒绝**。调用方（提交路径）必须
    据此抛错，而不是继续执行——只有预览路径才允许"带冲突返回"，以便
    前端在提交前把问题摆给用户看。
    """

    sequence: str
    name: str
    mutations: list[ManualMutation] = field(default_factory=list)
    blocked: list[BlockedTarget] = field(default_factory=list)
    rejected_substitutions: list[dict[str, Any]] = field(default_factory=list)
    locked_positions: list[int] = field(default_factory=list)
    target_positions: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def position_count(self) -> int:
        return len({item.position for item in self.mutations})

    def to_dict(self) -> dict[str, Any]:
        return {
            "length": len(self.sequence),
            "name": self.name,
            "target_positions": self.target_positions,
            "locked_positions": self.locked_positions,
            "mutation_count": len(self.mutations),
            "position_count": self.position_count,
            "mutations": [item.to_dict() for item in self.mutations],
            "blocked": [item.to_dict() for item in self.blocked],
            "rejected_substitutions": self.rejected_substitutions,
            "warnings": self.warnings,
        }

    # ----------------------------------------------------------------- #
    # 导出
    # ----------------------------------------------------------------- #

    def fasta_records(self, include_wild_type: bool = True) -> list[str]:
        """FASTA 记录列表；默认把野生型放在最前面作为基准对照。"""
        records: list[str] = []
        if include_wild_type:
            records.append(f">{self.name}_WT\n{self.sequence}")
        records.extend(item.fasta_record() for item in self.mutations)
        return records

    def to_fasta(self, include_wild_type: bool = True) -> str:
        return "\n".join(self.fasta_records(include_wild_type)) + "\n"

    #: CSV 列定义。前 3 列与需求要求的
    #: ``sequence_id, fasta, mutation_note`` 一致，其余为便于人工核对而附加。
    CSV_COLUMNS: tuple[str, ...] = (
        "sequence_id",
        "fasta",
        "mutation_note",
        "mutation",
        "position",
        "wild_type",
        "mutant",
        "sequence",
        "sequence_length",
    )

    def to_csv(self, include_wild_type: bool = True) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(self.CSV_COLUMNS)

        if include_wild_type:
            writer.writerow(
                [
                    f"{self.name}_WT",
                    f">{self.name}_WT\n{self.sequence}",
                    "野生型基准对照",
                    "",
                    "",
                    "",
                    "",
                    self.sequence,
                    len(self.sequence),
                ]
            )

        for item in self.mutations:
            writer.writerow(
                [
                    item.sequence_id,
                    item.fasta_record(),
                    item.note,
                    item.label,
                    item.display_position,
                    item.wild_type,
                    item.mutant,
                    item.sequence,
                    len(item.sequence),
                ]
            )
        return buffer.getvalue()


# --------------------------------------------------------------------------- #
# 规划
# --------------------------------------------------------------------------- #


def normalize_positions(values: Iterable[Any] | None) -> list[int]:
    """把人工输入的位点规整为**升序去重**的 1-based 整数列表。

    拒绝非整数输入而不是静默跳过：位点是人工手输的，写错必须让人看见。
    """
    if not values:
        return []
    result: set[int] = set()
    for raw in values:
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"位点必须是整数，收到 {raw!r}") from exc
        if value < 1:
            raise ValidationError(f"位点必须从 1 开始计数，收到 {value}")
        result.add(value)
    return sorted(result)


def _normalize_substitutions(
    substitutions: dict[str, list[str]] | None,
) -> dict[int, list[str]]:
    """``{“23”: [“A”,”S”]}`` -> ``{23: [“A”,”S”]}``，并校验氨基酸合法性。"""
    if not substitutions:
        return {}
    parsed: dict[int, list[str]] = {}
    for key, values in substitutions.items():
        try:
            position = int(key)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"substitutions 的位置键必须是整数，收到 {key!r}") from exc
        cleaned: list[str] = []
        for item in values or []:
            residue = str(item).strip().upper()
            if not residue:
                continue
            if len(residue) != 1 or residue not in AA_ORDER:
                raise ValidationError(
                    f"位点 {position} 的替换氨基酸非法: {item!r}",
                    detail={"available": list(AA_ORDER)},
                )
            if residue not in cleaned:
                cleaned.append(residue)
        parsed[position] = cleaned
    return parsed


def build_manual_plan(
    sequence: str,
    *,
    target_positions: Iterable[Any],
    substitutions: dict[str, list[str]] | None = None,
    locked_positions: Iterable[Any] | None = None,
    protein_type: str = "generic",
    name: str | None = None,
    notes: dict[str, str] | None = None,
    exclude_terminal: int | None = None,
) -> ManualPlan:
    """构建并**校验**人工突变清单。

    校验覆盖五类拒绝原因，全部逐条记录在 ``ManualPlan.blocked`` 中：

    ====================  ==================================================
    原因                 说明
    ====================  ==================================================
    超出序列范围         位点 < 1 或 > 序列长度
    锁定位点             人工声明的禁改位点（如二硫键 Cys）
    规则包保护位点       蛋白类型规则包给出的保护位点（含信号肽区）
    序列末端             位于两端 ``exclude_terminal`` 位内，突变会干扰翻译起始/终止
    非法替换             非标准氨基酸，或与野生型残基相同
    ====================  ==================================================

    Args:
        target_positions: 人工选定的目标位点（1-based）。
        substitutions: 每位点的候选氨基酸 ``{"23": ["A", "S"]}``；
            某位点未给出时使用 :func:`default_substitutions` 的默认集合。
        locked_positions: 人工声明的禁止突变位点（1-based）。
        exclude_terminal: 两端排除的残基数；留空取配置 ``scan.exclude_terminal``。
    """
    sequence = (sequence or "").strip().upper()
    if not sequence:
        raise ValidationError("序列不能为空")

    targets = normalize_positions(target_positions)
    if not targets:
        raise ValidationError("人工模式必须至少给出一个目标位点")

    locks = normalize_positions(locked_positions)
    parsed_substitutions = _normalize_substitutions(substitutions)
    length = len(sequence)

    terminal = (
        int(exclude_terminal)
        if exclude_terminal is not None
        else int(load_platform_config().get("scan", {}).get("exclude_terminal", 1))
    )

    # 规则包保护位点（0-based）。人工模式下**仍然生效**：
    # 专用规则包保护的催化残基、胶原 Gly 位等一旦被突变，结果没有业务意义，
    # 因此与其算出无效分数，不如在提交前明确挡住。
    protected = get_rulepack(protein_type).merged_protections(sequence)

    plan = ManualPlan(
        sequence=sequence,
        name=(name or "Protein").strip() or "Protein",
        locked_positions=locks,
        target_positions=targets,
    )

    seen_labels: set[str] = set()

    for display_position in targets:
        index = display_position - 1

        # ---------- 位点级校验 ----------
        if index < 0 or index >= length:
            plan.blocked.append(
                BlockedTarget(display_position, "", f"超出序列范围（序列长度 {length}）")
            )
            continue

        wild_type = sequence[index]

        if display_position in set(locks):
            plan.blocked.append(
                BlockedTarget(
                    display_position,
                    wild_type,
                    "该位点已被人工标记为锁定位点，禁止突变",
                )
            )
            continue

        if index in protected:
            plan.blocked.append(
                BlockedTarget(
                    display_position,
                    wild_type,
                    f"该位点受 {protein_type} 规则包保护：{protected[index]}",
                )
            )
            continue

        if index < terminal or index >= length - terminal:
            plan.blocked.append(
                BlockedTarget(
                    display_position,
                    wild_type,
                    f"位于序列末端 {terminal} 位内，突变会干扰翻译起始/终止",
                )
            )
            continue

        # ---------- 替换氨基酸 ----------
        mutants = parsed_substitutions.get(display_position)
        if mutants is None:
            mutants = list(default_substitutions(wild_type))

        for mutant in mutants:
            label = f"{wild_type}{display_position}{mutant}"
            if mutant == wild_type:
                plan.rejected_substitutions.append(
                    {
                        "position": display_position,
                        "mutant": mutant,
                        "reason": "与野生型残基相同，不构成突变",
                    }
                )
                continue
            if label in seen_labels:
                plan.rejected_substitutions.append(
                    {
                        "position": display_position,
                        "mutant": mutant,
                        "reason": "重复的突变标签，已跳过",
                    }
                )
                continue
            seen_labels.add(label)

            note = (notes or {}).get(label) or (notes or {}).get(str(display_position)) or ""
            plan.mutations.append(
                ManualMutation(
                    position=index,
                    wild_type=wild_type,
                    mutant=mutant,
                    label=label,
                    sequence=sequence[:index] + mutant + sequence[index + 1 :],
                    sequence_id=f"{plan.name}_Mut_{label}",
                    note=note,
                )
            )

    # ---------- 规模保护 ----------
    if len(plan.mutations) > MAX_MUTATIONS:
        raise ValidationError(
            f"本批人工突变共 {len(plan.mutations)} 条，超过单批上限 {MAX_MUTATIONS}。"
            f"需求建议单批 {RECOMMENDED_BATCH[0]}~{RECOMMENDED_BATCH[1]} 条"
            f"（含野生对照）以控制变量；请拆分为多批提交。",
            detail={"mutation_count": len(plan.mutations), "limit": MAX_MUTATIONS},
        )

    # ---------- 汇总提示 ----------
    if plan.blocked:
        plan.warnings.append(
            f"{len(plan.blocked)} 个目标位点被拒绝，明细见 blocked 字段。"
        )
    if plan.rejected_substitutions:
        plan.warnings.append(
            f"{len(plan.rejected_substitutions)} 条替换被跳过（与野生型相同或重复），明细见 rejected_substitutions 字段。"
        )
    if not plan.mutations:
        plan.warnings.append("没有任何有效的突变可评估，请检查目标位点与锁定位点设置。")

    return plan


def ensure_plan_is_runnable(plan: ManualPlan) -> None:
    """提交前校验：存在被拒绝的输入就抛错。

    自动模式可以静默降采样（有 ``downsampled`` 字段说明），但人工模式的位点是
    人一个一个指定的——任何一条被丢掉都说明"人给的和平台算的不是一回事"，
    属于必须让人看见的错误，因此这里直接拒绝执行。
    """
    if plan.blocked:
        detail = "; ".join(
            f"{item.position}{item.residue}: {item.reason}" if item.residue else f"{item.position}: {item.reason}"
            for item in plan.blocked
        )
        raise ValidationError(
            f"有 {len(plan.blocked)} 个目标位点无法突变，已拒绝提交：{detail}",
            detail={"blocked": [item.to_dict() for item in plan.blocked]},
        )
    if not plan.mutations:
        raise ValidationError(
            "没有有效的突变：请检查目标位点的替换氨基酸是否都与野生型相同",
            detail={"rejected_substitutions": plan.rejected_substitutions},
        )


def plan_catalog() -> dict[str, Any]:
    """供前端展示的人工模式默认值说明。"""
    return {
        "default_substitutions": {
            key: list(value) for key, value in DEFAULT_SUBSTITUTIONS.items()
        },
        "residue_classes": {
            "positive": sorted(CHARGE_POSITIVE),
            "negative": sorted(CHARGE_NEGATIVE),
            "aromatic": sorted(AROMATIC),
            "hydrophobic": sorted(HYDROPHOBIC),
            "polar": sorted(POLAR),
            "special": sorted(BACKBONE_SPECIAL),
        },
        "note": (
            "未指定候选氨基酸的位点，将按残基类别套用上表（每类 3~4 个，含丙氨酸扫描）。"
            "人工可完全覆盖，平台不猜测锁定位点。"
        ),
    }
