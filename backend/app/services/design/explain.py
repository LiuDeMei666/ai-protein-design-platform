"""多维度可解释评分卡合成。

需求文档明确要求"输出突变方案时提供可解释性评分（如稳定性贡献、活性影响、
折叠可行性等多维度打分），便于实验人员优先筛选"。

因此本模块的硬约束是：
**每一条候选都必须携带逐维度分数 + 中文依据 + 告警标签，禁止只给一个总分。**

评分维度（权重来自 ``configs/default.yaml`` 的 ``scoring.dimensions``）：

======================  ==========================================
维度                    含义（均为 0-100，越高越好）
======================  ==========================================
stability               稳定性贡献（ESM-2 掩码边缘似然比）
activity                活性影响（模型保守性 + 距功能位点距离 + 规则包偏好）
foldability             折叠可行性（pLDDT 先验 + 模型耐受度 + 二级结构）
expression              宿主表达适配（停滞基序、含硫残基、N 端规则变化）
risk                    新增风险位点（越安全越高；按罚项计入总分）
======================  ==========================================

总分 = Σ(正维度 w·s)/Σw + w_risk·(s_risk − 100)/Σw，
因此**各维度贡献之和精确等于总分**，前端瀑布图可零残差分解。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.config import load_platform_config
from ...core.logging import get_logger
from ..property.expression import count_stalling_motifs
from ..sequence.residue_properties import (
    RESIDUE_CHARGE_PH7,
    charge_change,
    is_conservative,
)
from .rulepacks import GenericRulePack, RuleContext, RulePack, get_generic_rulepack, merge_verdicts
from .scanner import PositionContext, build_combinatorial_sequence, mutation_label
from .scorers.biophys_delta import assess_mutation_risk
from .scorers.foldability import assess_foldability
from .scorers.zero_shot import ZeroShotScores, activity_score

logger = get_logger(__name__)

#: 正维度（分数越高越好，按权重加权求和）
POSITIVE_DIMENSIONS = ("stability", "activity", "foldability", "expression")
RISK_DIMENSION = "risk"


@dataclass
class ScoreDimension:
    """一个评分维度。"""

    key: str
    label: str
    raw: float
    weight: float
    contribution: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "raw": round(self.raw, 2),
            "weight": round(self.weight, 4),
            "contribution": round(self.contribution, 2),
            "rationale": self.rationale,
        }


@dataclass
class Candidate:
    """一条突变候选方案（含完整可解释评分）。"""

    mutations: list[str]
    positions: list[int]
    total_score: float
    dimensions: list[ScoreDimension]
    flags: list[str] = field(default_factory=list)
    category: str = "generic"
    rationale: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    site_count: int = 1

    def dimension(self, key: str) -> ScoreDimension | None:
        for item in self.dimensions:
            if item.key == key:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutations": self.mutations,
            "positions": [position + 1 for position in self.positions],  # 前端用 1-based
            "site_count": self.site_count,
            "total_score": round(self.total_score, 2),
            "dimensions": [item.to_dict() for item in self.dimensions],
            "flags": self.flags,
            "category": self.category,
            "rationale": self.rationale,
            "detail": self.detail,
        }


def _scoring_config() -> tuple[dict[str, dict[str, Any]], bool]:
    config = load_platform_config().get("scoring", {})
    dimensions = config.get("dimensions", {}) or {}
    return dimensions, bool(config.get("risk_is_penalty", True))


def _weight(key: str, default: float) -> float:
    dimensions, _ = _scoring_config()
    entry = dimensions.get(key, {})
    return float(entry.get("weight", default))


def _label(key: str, default: str) -> str:
    dimensions, _ = _scoring_config()
    entry = dimensions.get(key, {})
    return str(entry.get("label", default))


def assemble_dimensions(
    stability: tuple[float, str],
    activity: tuple[float, str],
    foldability: tuple[float, str],
    expression: tuple[float, str],
    risk: tuple[float, str],
) -> tuple[float, list[ScoreDimension]]:
    """把五个维度的分数合成为总分与可分解的贡献列表。"""
    _, risk_is_penalty = _scoring_config()

    values: dict[str, tuple[float, str]] = {
        "stability": stability,
        "activity": activity,
        "foldability": foldability,
        "expression": expression,
        RISK_DIMENSION: risk,
    }

    positive_weights = {
        key: _weight(key, default) for key, default in (
            ("stability", 0.28), ("activity", 0.24), ("foldability", 0.18), ("expression", 0.12)
        )
    }
    risk_weight = _weight(RISK_DIMENSION, 0.18)
    total_weight = sum(positive_weights.values()) + risk_weight
    if total_weight <= 0:
        total_weight = 1.0

    dimensions: list[ScoreDimension] = []
    total = 0.0

    for key in POSITIVE_DIMENSIONS:
        score, rationale = values[key]
        weight = positive_weights[key]
        contribution = weight * score / total_weight
        total += contribution
        dimensions.append(
            ScoreDimension(
                key=key,
                label=_label(key, key),
                raw=score,
                weight=weight / total_weight,
                contribution=contribution,
                rationale=rationale,
            )
        )

    risk_score, risk_rationale = values[RISK_DIMENSION]
    if risk_is_penalty:
        # 作为罚项：满安全分（100）不扣分，风险越大扣得越多
        contribution = risk_weight * (risk_score - 100.0) / total_weight
    else:
        contribution = risk_weight * risk_score / total_weight
    total += contribution
    dimensions.append(
        ScoreDimension(
            key=RISK_DIMENSION,
            label=_label(RISK_DIMENSION, "新增风险位点"),
            raw=risk_score,
            weight=risk_weight / total_weight,
            contribution=contribution,
            rationale=risk_rationale,
        )
    )

    return round(max(0.0, min(100.0, total)), 2), dimensions


def expression_delta_score(
    sequence: str,
    position: int,
    mutant: str,
    *,
    before_stall: float | None = None,
    before_sulfur: float | None = None,
) -> tuple[float, str]:
    """表达适配维度的变化分（基线 70，按变化加减）。

    Args:
        before_stall: 野生型停滞基序加权计数。批量评估时由调用方预计算传入，
            避免对每条候选重复扫描全序列。
        before_sulfur: 野生型 (Cys+Trp) 占比，同上。
    """
    mutant_sequence = sequence[:position] + mutant + sequence[position + 1 :]
    length = max(1, len(sequence))

    if before_stall is None:
        before_stall, _ = count_stalling_motifs(sequence)
    after_stall, _ = count_stalling_motifs(mutant_sequence)

    if before_sulfur is None:
        before_sulfur = (sequence.count("C") + sequence.count("W")) / length
    after_sulfur = (mutant_sequence.count("C") + mutant_sequence.count("W")) / length

    score = 70.0
    notes: list[str] = []

    stall_delta = after_stall - before_stall
    if stall_delta > 0:
        score -= 8.0 * stall_delta
        notes.append(f"新增核糖体停滞敏感基序（+{stall_delta:.0f}）")
    elif stall_delta < 0:
        score += 6.0 * abs(stall_delta)
        notes.append(f"消除核糖体停滞敏感基序（{stall_delta:.0f}）")

    sulfur_delta = after_sulfur - before_sulfur
    if abs(sulfur_delta) > 1e-6:
        score -= sulfur_delta * 300.0
        notes.append(
            f"含硫/吲哚残基占比变化 {sulfur_delta:+.4f}"
            + ("（合成成本上升）" if sulfur_delta > 0 else "（合成成本下降）")
        )

    # N 端规则：只有 Met 后第二位残基有意义
    if position == 1:
        from ..property.biophys import N_END_RULE

        before_class = N_END_RULE.get(sequence[1], "unknown")
        after_class = N_END_RULE.get(mutant, "unknown")
        mapping = {"stabilizing": 95.0, "neutral": 70.0, "destabilizing": 35.0, "unknown": 55.0}
        score = mapping.get(after_class, 55.0)
        notes.append(
            f"N 端规则由 {before_class} 变为 {after_class}"
            + ("（半衰期延长，表达量有望提升）" if after_class == "stabilizing" else "")
        )

    score = max(0.0, min(100.0, score))
    rationale = "；".join(notes) if notes else "该替换不改变停滞基序与含硫残基负荷，表达影响中性"
    return round(score, 2), rationale


def evaluate_single(
    sequence: str,
    position: int,
    mutant: str,
    zero_shot: ZeroShotScores,
    *,
    context: PositionContext | None = None,
    rulepack: RulePack | None = None,
    generic_pack: GenericRulePack | None = None,
    protein_type: str = "generic",
    wild_profile: list[float] | None = None,
    before_stall: float | None = None,
    before_sulfur: float | None = None,
) -> Candidate:
    """对单个突变做完整五维评分。

    ``wild_profile`` / ``before_stall`` / ``before_sulfur`` 是**野生型的预计算量**。
    批量评估（一次扫描上万个候选）时必须传入，否则会对每条候选重复做
    全序列级别的 O(L) 计算，实测会带来数十倍的时间浪费。
    """
    wild_type = sequence[position]
    generic_pack = generic_pack or get_generic_rulepack()
    rulepack = rulepack or generic_pack

    # ---------- 稳定性 ----------
    delta_logp = zero_shot.delta(position, mutant)
    stability_score = zero_shot.stability_score(position, mutant)
    if stability_score is None:
        stability_score = 50.0
    wt_logp = zero_shot.wt_logp(position)

    if delta_logp is None:
        stability_rationale = f"ESM-2 未返回 {wild_type}->{mutant} 的打分结果（该位点可能超出模型窗口）"
    else:
        verdict_text = (
            "模型认可该替换"
            if delta_logp > -1.0
            else "模型对该替换略有不认可" if delta_logp > -3.0 else "模型对该替换存在明显抵触"
        )
        stability_rationale = (
            f"ESM-2 掩码边缘似然比 ΔlogP={delta_logp:.2f}（{wild_type}->{mutant}）：{verdict_text}"
        )
        if wt_logp is not None:
            stability_rationale += f"；该位点野生型掩码对数概率 {wt_logp:.2f}"

    # ---------- 活性 / 功能 ----------
    rule_context = RuleContext(
        sequence=sequence,
        position=position,
        wild_type=wild_type,
        mutant=mutant,
        context=context,
    )
    verdict = merge_verdicts(rulepack.activity_factor(rule_context), generic_pack.activity_factor(rule_context))

    # 距离功能位点的折扣：规则包通过 evidence 回传距离信息
    distance = verdict.evidence.get("distance_to_catalytic_ser")
    if distance is None:
        proximity_factor = 1.0
    elif distance <= 4:
        proximity_factor = 0.1
    elif distance <= 8:
        proximity_factor = 0.55
    elif distance <= 15:
        proximity_factor = 0.85
    else:
        proximity_factor = 1.0

    activity_value, activity_desc = activity_score(
        delta_logp=delta_logp,
        wt_logprob=wt_logp,
        proximity_factor=proximity_factor,
        rulepack_factor=verdict.factor,
    )
    activity_rationale = f"{activity_desc}。{verdict.note}" if verdict.note else activity_desc

    # ---------- 折叠可行性 ----------
    foldability_result = assess_foldability(
        sequence, position, mutant, float(stability_score), context
    )

    # ---------- 表达适配 ----------
    expression_value, expression_rationale = expression_delta_score(
        sequence,
        position,
        mutant,
        before_stall=before_stall,
        before_sulfur=before_sulfur,
    )

    # ---------- 新增风险位点 ----------
    risk_result = assess_mutation_risk(
        sequence, position, mutant, context, protein_type, wild_profile=wild_profile
    )

    total, dimensions = assemble_dimensions(
        stability=(float(stability_score), stability_rationale),
        activity=(activity_value, activity_rationale),
        foldability=(foldability_result.score, foldability_result.rationale),
        expression=(expression_value, expression_rationale),
        risk=(risk_result.safety_score, "；".join(risk_result.flags) or "未引入新的化学不稳定位点"),
    )

    flags = list(dict.fromkeys([*verdict.flags, *risk_result.flags]))

    return Candidate(
        mutations=[mutation_label(sequence, position, mutant)],
        positions=[position],
        total_score=total,
        dimensions=dimensions,
        flags=flags,
        category=protein_type,
        rationale=(
            f"单点突变 {mutation_label(sequence, position, mutant)}："
            f"稳定性 {stability_score:.0f}、活性 {activity_value:.0f}、"
            f"折叠 {foldability_result.score:.0f}、表达 {expression_value:.0f}、"
            f"安全性 {risk_result.safety_score:.0f}。"
            + (f" 告警：{'；'.join(flags)}。" if flags else " 未触发风险告警。")
        ),
        detail={
            "delta_logp": round(delta_logp, 3) if delta_logp is not None else None,
            "wt_logprob": round(wt_logp, 3) if wt_logp is not None else None,
            "rulepack_note": verdict.note,
            "rulepack_factor": round(verdict.factor, 3),
            "risk_details": risk_result.details,
            "deltas": risk_result.deltas,
            "foldability_evidence": foldability_result.evidence,
            "context": context.describe() if context else "无结构信息",
            "is_conservative": is_conservative(wild_type, mutant),
            "charge_change": round(charge_change(wild_type, mutant), 2),
        },
        site_count=1,
    )


def refine_combination(
    sequence: str,
    mutations: dict[int, str],
    *,
    context_map: dict[int, PositionContext] | None = None,
    rulepack: RulePack | None = None,
    generic_pack: GenericRulePack | None = None,
    protein_type: str = "generic",
) -> tuple[float, list[dict[str, Any]]]:
    """对组合突变做**上下文依赖**的精确重打分。

    对组合中的每个突变 (p, m)，构造"其他突变已应用、但 p 位保持野生型"的序列，
    在该背景下掩码 p 并读取 ΔlogP。这样得到的是真正考虑了上位效应的效应值。
    """
    from ..embedding.esm2 import get_service

    service = get_service()
    service.ensure_loaded()

    context_map = context_map or {}
    generic_pack = generic_pack or get_generic_rulepack()
    rulepack = rulepack or generic_pack

    combined = build_combinatorial_sequence(sequence, mutations)
    dimensions: list[ScoreDimension] = []
    per_site: list[dict[str, Any]] = []

    stability_values: list[float] = []
    activity_values: list[float] = []
    foldability_values: list[float] = []
    expression_values: list[float] = []
    safety_values: list[float] = []

    for position, mutant in sorted(mutations.items()):
        # 背景序列：组合序列中把该位点还原为野生型
        background = combined[:position] + sequence[position] + combined[position + 1 :]
        try:
            result = service.masked_marginal(background, positions=[position])
            from .scorers.zero_shot import normalize_delta

            row = 0
            aa_index = result.aa_order.index(mutant)
            delta = float(result.delta_logprob[row, aa_index])
            normalized = float(normalize_delta(result.delta_logprob[row : row + 1, :])[0, aa_index])
            wt_logp = float(result.wt_logprob[row])
        except Exception as exc:
            logger.warning("组合背景下位点 %d 打分失败: %s", position, exc)
            delta, normalized, wt_logp = None, 50.0, None

        context = context_map.get(position)
        rule_context = RuleContext(
            sequence=sequence,
            position=position,
            wild_type=sequence[position],
            mutant=mutant,
            context=context,
        )
        verdict = merge_verdicts(
            rulepack.activity_factor(rule_context), generic_pack.activity_factor(rule_context)
        )

        distance = verdict.evidence.get("distance_to_catalytic_ser")
        proximity = 1.0 if distance is None else (0.1 if distance <= 4 else 0.55 if distance <= 8 else 0.85)

        activity_value, activity_desc = activity_score(
            delta_logp=delta, wt_logprob=wt_logp, proximity_factor=proximity, rulepack_factor=verdict.factor
        )

        risk_result = assess_mutation_risk(sequence, position, mutant, context, protein_type)
        expression_value, _ = expression_delta_score(sequence, position, mutant)

        # 折叠可行性必须走与单点评估**同一个函数**，否则同一突变在
        # "单点列表"与"组合列表"中会得到不同分数（实测差 6 分），破坏可信度。
        foldability_result = assess_foldability(sequence, position, mutant, normalized, context)

        stability_values.append(normalized)
        activity_values.append(activity_value)
        safety_values.append(risk_result.safety_score)
        expression_values.append(expression_value)
        foldability_values.append(foldability_result.score)

        per_site.append(
            {
                "mutation": mutation_label(sequence, position, mutant),
                "position": position + 1,
                "delta_logp_in_context": round(delta, 3) if delta is not None else None,
                "stability": round(normalized, 2),
                "activity": round(activity_value, 2),
                "safety": risk_result.safety_score,
                "flags": risk_result.flags,
            }
        )

    count = max(1, len(mutations))
    # 组合的维度分取各站点的**平均值**，避免位点数越多分越高（否则组合永远优于单点）
    stability = sum(stability_values) / count
    activity = sum(activity_values) / count
    foldability = sum(foldability_values) / count
    expression = sum(expression_values) / count
    safety = min(safety_values) if safety_values else 50.0  # 安全性取最差站点

    total, dimension_objects = assemble_dimensions(
        stability=(stability, f"{count} 个位点的上下文依赖平均稳定性分（含上位效应）"),
        activity=(activity, f"{count} 个位点的平均活性影响分"),
        foldability=(foldability, f"{count} 个位点的平均折叠可行性（以上下文依赖耐受度近似）"),
        expression=(expression, f"{count} 个位点的平均表达适配分"),
        risk=(safety, "取各站点中最低的安全分（最差位点决定整体风险）"),
    )

    return total, [item.to_dict() for item in dimension_objects] + [{"per_site": per_site}]


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """按总分降序排序，并给出排名理由。"""
    ordered = sorted(candidates, key=lambda item: -item.total_score)
    for index, item in enumerate(ordered, start=1):
        item.detail["rank"] = index
    return ordered
