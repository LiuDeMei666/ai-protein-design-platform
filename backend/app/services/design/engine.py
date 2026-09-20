"""智能突变设计引擎（对外主入口）。

流程::

    1. 规则包加载（按蛋白类型）
    2. 扫描计划（排除末端 / 保护位点 / 遵守区段与位点上限）
    3. ESM-2 零样本打分（全位点 × 20 氨基酸，批量掩码前向）
    4. 逐候选五维可解释评分
    5. [组合模式] 束搜索 + 上下文依赖精评
    6. 排序、汇总、输出改造策略建议

支持三种模式（对应需求文档"单点突变、多点组合突变及局部序列优化"）：

* ``single``      —— 输出 Top-N 单点突变候选
* ``combination`` —— 在单点基础上做组合搜索（含上位效应纠正）
* ``local``       —— 限定区段内只做**保守替换**（体积等级与极性不变），
  适合"不想大改、只想微调"的场景
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...core.config import load_platform_config
from ...core.errors import SequenceError, ValidationError
from ...core.logging import describe_sequence, get_logger
from ..property.aggregation import aggregation_profile
from ..property.expression import count_stalling_motifs
from ..sequence.residue_properties import is_conservative
from .combiner import SingleMutation, search_combinations
from .explain import Candidate, evaluate_single, rank_candidates, refine_combination
from .rulepacks import RulePack, get_generic_rulepack, get_rulepack, rulepack_catalog
from .scanner import (
    AA_ORDER,
    PositionContext,
    build_position_contexts,
    build_scan_plan,
    mutation_label,
)
from .scorers.zero_shot import compute_zero_shot, summarize_scores

logger = get_logger(__name__)

MODES: tuple[str, ...] = ("single", "combination", "local", "manual")


@dataclass
class DesignRequest:
    """突变设计请求。"""

    sequence: str
    protein_type: str = "generic"
    mode: str = "single"
    host_system: str = "ecoli"
    region: tuple[int, int] | None = None
    max_positions: int | None = None
    top_n: int = 50
    max_sites: int = 3
    beam_width: int = 8
    structure_stats: dict[str, Any] | None = None
    exclude_terminal: int | None = None

    # ---------- 人工模式（mode="manual"）专用 ----------
    #: 人工选定的目标位点（1-based）。自动模式忽略此字段。
    target_positions: list[int] | None = None
    #: 每位点的候选氨基酸 ``{"23": ["A", "S"]}``；某位点缺省时按残基类别套用默认集合。
    substitutions: dict[str, list[str]] | None = None
    #: 人工声明的禁止突变位点（1-based，如二硫键 Cys）。平台不做自动识别。
    locked_positions: list[int] | None = None
    #: 突变备注，键为突变标签（如 ``"W23A"``），导出 CSV 时写入 mutation_note 列。
    mutation_notes: dict[str, str] | None = None
    #: 序列名，用于生成导出的唯一 sequence_id（如 ``Hevb6_Mut_W23A``）。
    name: str | None = None


@dataclass
class DesignResult:
    """突变设计结果。

    ``candidates`` 内部保留**全部**候选（组合搜索需要更大的候选池），
    但对外序列化时按 ``top_n`` 截断。这一点很关键：
    一次全长胶原蛋白扫描会产生 2 万条以上候选，若把全部候选塞进作业结果 JSON，
    既违反接口契约（用户只要 10 条），又会把数据库撑爆。
    """

    sequence: str
    protein_type: str
    mode: str
    candidates: list[Candidate] = field(default_factory=list)
    combinations: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    hints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)
    rulepack: dict[str, Any] = field(default_factory=dict)
    zero_shot_stats: dict[str, Any] = field(default_factory=dict)
    #: 人工模式专有信息：目标位点、锁定位点、突变清单与规模。
    #: 自动模式为空字典。前端据此展示"人给的和平台算的是否一致"。
    manual: dict[str, Any] = field(default_factory=dict)
    #: 对外返回的候选数量上限（来自请求的 top_n）
    top_n: int = 50

    def to_dict(self, top_n: int | None = None) -> dict[str, Any]:
        limit = self.top_n if top_n is None else top_n
        candidates = self.candidates[:limit]
        return {
            "length": len(self.sequence),
            "protein_type": self.protein_type,
            "mode": self.mode,
            "plan": self.plan,
            "rulepack": self.rulepack,
            "manual": self.manual,
            "candidates": [item.to_dict() for item in candidates],
            "returned_candidates": len(candidates),
            "total_candidates": len(self.candidates),
            "combinations": self.combinations,
            "summary": self.summary,
            "hints": self.hints,
            "warnings": self.warnings,
            "zero_shot_stats": self.zero_shot_stats,
        }


def _build_summary(candidates: list[Candidate], combinations: list[dict[str, Any]]) -> dict[str, Any]:
    """设计结果汇总。"""
    if not candidates:
        return {"candidate_count": 0, "combination_count": len(combinations)}

    scores = np.array([item.total_score for item in candidates], dtype=np.float64)
    top = candidates[0]
    flagged = sum(1 for item in candidates if item.flags)

    # 统计最常被推荐的位点（突变热点）
    hot_positions: dict[int, int] = {}
    for item in candidates[: max(10, len(candidates) // 5)]:
        for position in item.positions:
            hot_positions[position + 1] = hot_positions.get(position + 1, 0) + 1
    top_hotspots = sorted(hot_positions.items(), key=lambda pair: -pair[1])[:10]

    return {
        "candidate_count": len(candidates),
        "combination_count": len(combinations),
        "score_max": round(float(scores.max()), 2),
        "score_mean": round(float(scores.mean()), 2),
        "score_median": round(float(np.median(scores)), 2),
        "flagged_count": flagged,
        "top_candidate": {
            "mutations": top.mutations,
            "total_score": round(top.total_score, 2),
            "rationale": top.rationale,
        },
        "hotspots": [{"position": position, "count": count} for position, count in top_hotspots],
    }


def _validate_request(request: DesignRequest) -> None:
    if request.mode not in MODES:
        raise ValidationError(
            f"未知的设计模式: {request.mode}", detail={"available": list(MODES)}
        )
    if request.top_n <= 0:
        raise ValidationError("top_n 必须为正整数")
    if request.max_sites < 1:
        raise ValidationError("max_sites 必须 >= 1")


def run_design(request: DesignRequest) -> DesignResult:
    """执行一次突变设计。"""
    _validate_request(request)
    started = time.perf_counter()

    rulepack: RulePack = get_rulepack(request.protein_type)
    generic_pack = get_generic_rulepack()

    # ---------- 人工模式：位点与替换全部由人给出 ----------
    if request.mode == "manual":
        return _run_manual_design(request, rulepack, generic_pack, started)

    # ---------- 1) 扫描计划 ----------
    # 必须用 merged_protections：它在规则包的专用保护之上，再叠加通用的
    # 信号肽区保护（分泌型蛋白的 N 端信号肽会被切除，在该处推荐突变无意义）。
    protected = rulepack.merged_protections(request.sequence)
    plan = build_scan_plan(
        request.sequence,
        region=request.region,
        max_positions=request.max_positions,
        protected=protected,
        strict=True,
        exclude_terminal=request.exclude_terminal,
    )

    if not plan.positions:
        raise SequenceError(
            "没有任何可突变位点：请检查区段设置与保护规则"
            "（例如胶原蛋白的全部 Gly 位都被保护时，可突变位点会显著减少）"
        )

    sequence = plan.sequence
    context_map: dict[int, PositionContext] = build_position_contexts(sequence, request.structure_stats)

    # ---------- 2) 零样本打分 ----------
    zero_shot = compute_zero_shot(sequence, positions=plan.positions)
    zero_shot_stats = summarize_scores(zero_shot)

    # ---------- 3) 预计算野生型全序列量（避免逐候选重复计算） ----------
    wild_profile = aggregation_profile(sequence, window=7)
    before_stall, _ = count_stalling_motifs(sequence)
    before_sulfur = (sequence.count("C") + sequence.count("W")) / max(1, len(sequence))

    # ---------- 4) 逐候选评分 ----------
    candidates: list[Candidate] = []
    skipped_conservative = 0
    for position in zero_shot.positions:
        wild_type = sequence[position]
        row = zero_shot.index_of(position)
        if row is None:
            continue
        for column, mutant in enumerate(AA_ORDER):
            if mutant == wild_type:
                continue
            # 局部优化模式：只保留保守替换
            if request.mode == "local" and not is_conservative(wild_type, mutant):
                skipped_conservative += 1
                continue
            candidate = evaluate_single(
                sequence,
                position,
                mutant,
                zero_shot,
                context=context_map.get(position),
                rulepack=rulepack,
                generic_pack=generic_pack,
                protein_type=request.protein_type,
                wild_profile=wild_profile,
                before_stall=before_stall,
                before_sulfur=before_sulfur,
            )
            candidates.append(candidate)

    candidates = rank_candidates(candidates)
    warnings = list(plan.notes)

    if request.mode == "local":
        warnings.append(
            f"局部优化模式仅评估保守替换（体积等级与极性均不变），"
            f"已跳过 {skipped_conservative} 个非保守候选。"
        )

    # ---------- 5) 组合突变 ----------
    combinations: list[dict[str, Any]] = []
    if request.mode in ("combination",) or (request.mode == "local" and request.max_sites > 1):
        pool_size = max(6, request.beam_width * 3)
        singles = [
            SingleMutation(
                position=item.positions[0],
                mutant=item.mutations[0][-1],
                score=item.total_score,
                label=item.mutations[0],
            )
            for item in candidates[:pool_size]
        ]

        def _refine(mutations: dict[int, str], _candidate) -> float:
            total, _details = refine_combination(
                sequence,
                mutations,
                context_map=context_map,
                rulepack=rulepack,
                generic_pack=generic_pack,
                protein_type=request.protein_type,
            )
            return total

        ranked_combinations = search_combinations(
            singles,
            max_sites=request.max_sites,
            beam_width=request.beam_width,
            pool_size=pool_size,
            refine_top_k=12,
            refine_fn=_refine,
        )
        combinations = [item.to_dict() for item in ranked_combinations]

    elapsed = time.perf_counter() - started
    summary = _build_summary(candidates, combinations)
    summary["elapsed_seconds"] = round(elapsed, 2)
    summary["evaluated_candidates"] = len(candidates)

    logger.info(
        "突变设计完成 %s type=%s mode=%s 候选=%d 组合=%d 耗时=%.1fs",
        describe_sequence(sequence),
        request.protein_type,
        request.mode,
        len(candidates),
        len(combinations),
        elapsed,
    )

    return DesignResult(
        sequence=sequence,
        protein_type=request.protein_type,
        mode=request.mode,
        candidates=candidates,
        combinations=combinations,
        summary=summary,
        hints=rulepack.design_hints(sequence),
        warnings=warnings,
        plan=plan.to_dict(),
        rulepack=rulepack.describe(),
        zero_shot_stats=zero_shot_stats,
        top_n=request.top_n,
    )


def _run_manual_design(
    request: DesignRequest,
    rulepack: RulePack,
    generic_pack: Any,
    started: float,
) -> DesignResult:
    """人工模式：只评估人工指定的单点突变。

    与自动模式的三点差异（全部是刻意的）：

    1. **不做位点扫描**——位点就是人工给的那些，不降采样、不预筛。
    2. **不做全 19 种替换**——只评估人工声明的候选氨基酸。
    3. **不做组合搜索**——需求要求"单点突变优先，减少变量，方便判断评估结果"；
       组合搜索会引入上位效应，反而干扰 MVP 阶段对评估链路的验证。

    评分链路**完全复用**自动模式：ESM-2 零样本掩码边缘 + 生物物理 Δ + 规则包。
    因此人工模式的结果与自动模式在数值上可直接比较——这一点很重要，
    需求第 3 条验收标准（"相同序列重复提交，打分是否稳定"）依赖于此。
    """
    from .manual import build_manual_plan, ensure_plan_is_runnable

    manual_plan = build_manual_plan(
        request.sequence,
        target_positions=request.target_positions or [],
        substitutions=request.substitutions,
        locked_positions=request.locked_positions,
        protein_type=request.protein_type,
        name=request.name,
        notes=request.mutation_notes,
        exclude_terminal=request.exclude_terminal,
    )
    # 人工模式下任何被拒绝的输入都直接终止：位点是人一个个指定的，
    # 丢掉任何一条都会让"人以为在评的"与"平台实际评的"不一致。
    ensure_plan_is_runnable(manual_plan)

    sequence = manual_plan.sequence
    positions = sorted({item.position for item in manual_plan.mutations})
    context_map: dict[int, PositionContext] = build_position_contexts(
        sequence, request.structure_stats
    )

    zero_shot = compute_zero_shot(sequence, positions=positions)
    zero_shot_stats = summarize_scores(zero_shot)

    # 野生型预计算量：与自动模式同一套口径，确保两条路径分数可比
    wild_profile = aggregation_profile(sequence, window=7)
    before_stall, _ = count_stalling_motifs(sequence)
    before_sulfur = (sequence.count("C") + sequence.count("W")) / max(1, len(sequence))

    candidates: list[Candidate] = []
    skipped_missing = 0
    for item in manual_plan.mutations:
        if zero_shot.index_of(item.position) is None:
            # 该位点未被零样本打分覆盖（例如超出模型窗口）。如实计数，
            # 不静默丢弃——人工指定的突变少一条都必须让人知道。
            skipped_missing += 1
            continue
        candidate = evaluate_single(
            sequence,
            item.position,
            item.mutant,
            zero_shot,
            context=context_map.get(item.position),
            rulepack=rulepack,
            generic_pack=generic_pack,
            protein_type=request.protein_type,
            wild_profile=wild_profile,
            before_stall=before_stall,
            before_sulfur=before_sulfur,
        )
        # 原始 ΔlogP 无需在这里补充：``evaluate_single`` 已经把它写进
        # ``detail["delta_logp"]``（连带 ``detail["wt_logprob"]``），
        # 与自动模式用的是同一份字段，前端可以直接取。
        candidates.append(candidate)

    candidates = rank_candidates(candidates)
    elapsed = time.perf_counter() - started
    summary = _build_summary(candidates, [])
    summary["elapsed_seconds"] = round(elapsed, 2)
    summary["evaluated_candidates"] = len(candidates)

    warnings = list(manual_plan.warnings)
    warnings.append(
        "人工模式：仅评估人工指定的单点突变，不做全位点扫描与组合搜索。"
        "评分口径与自动模式完全一致（ESM-2 零样本 + 生物物理 Δ + 规则包），"
        "其中稳定性维度本身就是相对野生型的 ΔlogP，因此天然是“相对野生对比”读数。"
    )
    if skipped_missing:
        warnings.append(
            f"{skipped_missing} 条人工指定的突变超出 ESM-2 打分窗口，未参与评估。"
        )

    manual_block = manual_plan.to_dict()
    manual_block["positions_zero_based"] = positions
    manual_block["evaluated"] = len(candidates)
    manual_block["skipped"] = skipped_missing

    logger.info(
        "人工突变评估完成 %s targets=%d 锁定位点=%d 突变=%d 已评估=%d 跳过=%d 耗时=%.1fs",
        describe_sequence(sequence),
        len(manual_plan.target_positions),
        len(manual_plan.locked_positions),
        len(manual_plan.mutations),
        len(candidates),
        skipped_missing,
        elapsed,
    )

    return DesignResult(
        sequence=sequence,
        protein_type=request.protein_type,
        mode="manual",
        candidates=candidates,
        combinations=[],
        summary=summary,
        hints=rulepack.design_hints(sequence),
        warnings=warnings,
        plan={
            "position_count": manual_plan.position_count,
            # 与自动模式不同：这里不是"位点数 × 19"，而是人工实际指定的突变条数
            "candidate_count": len(manual_plan.mutations),
            "positions": positions,
            "exclusions": [item.to_dict() for item in manual_plan.blocked],
            "protected": {},
            "downsampled": False,
            "downsample_note": "",
            "notes": manual_plan.warnings,
            "strategy": "manual",
        },
        rulepack=rulepack.describe(),
        zero_shot_stats=zero_shot_stats,
        manual=manual_block,
        # 人工指定的突变**一条都不能被截断**：人给的是确切的清单，
        # 不像自动模式那样"从海量候选里挑前 N 条"。因此上限取两者较大值。
        top_n=max(request.top_n, len(manual_plan.mutations)),
    )


def design_catalog() -> dict[str, Any]:
    """设计能力的元信息（供前端参数面板与文档使用）。"""
    from .manual import plan_catalog

    config = load_platform_config()
    return {
        "modes": [
            {"key": "single", "label": "单点突变", "description": "输出 Top-N 单点突变候选"},
            {"key": "combination", "label": "多点组合突变", "description": "束搜索 + 上位效应精评"},
            {"key": "local", "label": "局部序列优化", "description": "限定区段内只做保守替换"},
            {
                "key": "manual",
                "label": "人工指定突变",
                "description": "位点、替换氨基酸与锁定位点全部人工指定，平台只做评分（MVP 验证用）",
            },
        ],
        "protein_types": [
            {"key": key, "label": value.get("label", key), "description": value.get("description", "")}
            for key, value in rulepack_catalog().items()
        ],
        "scoring_dimensions": config.get("scoring", {}).get("dimensions", {}),
        "combination": config.get("combination", {}),
        "scan": config.get("scan", {}),
        "manual": plan_catalog(),
    }
