#!/usr/bin/env python
"""标准化测试案例 ①：胶原蛋白（人 I 型胶原 α1 链 COL1A1）。

案例目标（对应需求文档"针对胶原蛋白的标准化测试案例及验证报告"）
----------------------------------------------------------------
1. 验证平台能处理**超长序列**（COL1A1 前体 1464 aa），且不静默截断；
2. 验证胶原规则包能按 Gly-X-Y 周期性正确保护三股螺旋 Gly 位；
3. 验证羟脯氨酸位点优化与交联位点设计的推荐是否符合胶原领域知识；
4. 输出可用于验证报告的结构化定量结果。

重要工程事实
------------
COL1A1 前体远超 ESM Atlas 在线接口的 400 残基上限，因此平台会自动分片折叠。
**分片之间的相对空间取向未经建模**，因此本案例对三股螺旋的评估依赖
「序列周期性 + 规则包知识」，而不是拼接后的整体构象——这一点会在报告中显式声明。

用法::

    python scripts/run_case_collagen.py
    python scripts/run_case_collagen.py --mode local --region 180 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SEED = PROJECT_ROOT / "data" / "seeds" / "collagen_col1a1_human.fasta"
OUTPUT_DIR = PROJECT_ROOT / "data" / "validation"


def load_seed() -> tuple[str, str]:
    """读取种子序列，返回 (名称, 序列)。"""
    if not SEED.exists():
        raise SystemExit(
            f"缺少种子序列 {SEED}\n请先执行: python scripts/fetch_seed_sequences.py"
        )
    header = ""
    chunks: list[str] = []
    for line in SEED.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            header = header or line[1:].strip()
        else:
            chunks.append(line.strip())
    sequence = "".join(chunks).upper()
    if not sequence:
        raise SystemExit(f"种子文件 {SEED} 中没有序列内容")
    return header or "COL1A1", sequence


def main() -> int:
    parser = argparse.ArgumentParser(description="胶原蛋白标准化测试案例")
    parser.add_argument("--mode", default="single", choices=["single", "local"], help="设计模式")
    parser.add_argument("--region", nargs=2, type=int, default=None, help="限定区段（1-based 闭区间）")
    parser.add_argument("--skip-structure", action="store_true", help="跳过结构预测（省时）")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    from backend.app.core.logging import get_logger, setup_logging
    from backend.app.services.design.engine import DesignRequest, run_design
    from backend.app.services.design.rulepacks import get_rulepack
    from backend.app.services.sequence.feature_utils import (
        collagen_repeat_density,
        gly_xy_phase,
    )
    from backend.app.services.sequence.validator import validate_sequence

    setup_logging()
    logger = get_logger("case.collagen")

    name, sequence = load_seed()
    print("=" * 78)
    print(" 标准化测试案例 ① 胶原蛋白")
    print("=" * 78)
    print(f" 对象       : {name}")
    print(f" 序列长度   : {len(sequence)} aa")
    print(f" 设计模式   : {args.mode}")

    report: dict = {
        "case": "collagen",
        "target": name,
        "sequence_length": len(sequence),
        "mode": args.mode,
    }

    # ---------- 1) 序列与周期性分析 ----------
    check = validate_sequence(sequence)
    print(f" 序列校验   : {'通过' if check.ok else '失败'}（警告 {len(check.warnings)} 条）")
    report["sequence_check"] = {
        "ok": check.ok,
        "length": check.length,
        "warnings": check.warnings,
        "errors": check.errors,
    }

    phase_counts = {0: 0, 1: 0, 2: 0, "none": 0}
    hyp_sites_native = 0
    crosslink_native = 0
    for index, residue in enumerate(sequence):
        phase = gly_xy_phase(sequence, index)
        if phase is None:
            phase_counts["none"] += 1
            continue
        phase_counts[phase] += 1
        if phase == 2 and residue == "P":
            hyp_sites_native += 1
        if phase == 2 and residue == "K":
            crosslink_native += 1

    density = collagen_repeat_density(sequence)
    print(f" Gly-X-Y    : Gly位 {phase_counts[0]} · X位 {phase_counts[1]} · Y位 {phase_counts[2]}"
          f" · 非重复区 {phase_counts['none']}")
    print(f" 重复密度   : {density:.3f}")
    print(f" 天然羟化位点（Y位Pro）: {hyp_sites_native} 个")
    print(f" 天然交联位点（Y位Lys）: {crosslink_native} 个")

    report["collagen_periodicity"] = {
        "phase_counts": {str(key): value for key, value in phase_counts.items()},
        "repeat_density": density,
        "native_hyp_sites": hyp_sites_native,
        "native_crosslink_sites": crosslink_native,
    }

    # ---------- 2) 规则包保护位点 ----------
    rulepack = get_rulepack("collagen")
    protected = rulepack.protected_positions(sequence)
    gly_total = sum(1 for residue in sequence if residue == "G")
    print(f" 保护位点   : {len(protected)} 个（序列中共 {gly_total} 个 Gly，"
          f"其中处于三股螺旋相位的被保护）")
    report["rulepack"] = {
        "protected_count": len(protected),
        "gly_total": gly_total,
        "protection_rate": round(len(protected) / max(1, gly_total), 4),
    }

    # ---------- 3) 结构预测（可选） ----------
    structure_stats: dict | None = None
    if not args.skip_structure:
        from backend.app.services.structure.registry import predict_structure

        print("\n 结构预测中（超长序列会自动分片，可能需要数分钟）…")
        started = time.time()
        try:
            structure = predict_structure(sequence)
            elapsed = time.time() - started
            truncated_note = "分片预测" if len(structure.segments) > 1 else "单次预测"
            print(f" 结构来源   : {structure.source}（{truncated_note}，耗时 {elapsed:.1f}s）")
            print(f" 片段数     : {len(structure.segments)}")
            print(f" 平均 pLDDT : {structure.mean_plddt:.2f}")
            for warning in structure.warnings:
                print(f"   ⚠ {warning}")
            structure_stats = structure.stats
            report["structure"] = {
                "source": structure.source,
                "segments": [list(segment) for segment in structure.segments],
                "fragment_count": len(structure.segments),
                "mean_plddt": structure.mean_plddt,
                "truncated": structure.truncated,
                "warnings": structure.warnings,
                "elapsed_seconds": round(elapsed, 2),
                "degradation_reason": structure.degradation_reason,
            }
        except Exception as exc:
            print(f" 结构预测失败（案例将继续）：{exc}")
            report["structure"] = {"error": str(exc)}
    else:
        print("\n 已跳过结构预测")

    # ---------- 4) 突变设计 ----------
    region = tuple(args.region) if args.region else None
    print(f"\n 突变设计中（mode={args.mode}, 区段={region or '全长'}）…")
    started = time.time()
    request = DesignRequest(
        sequence=sequence,
        protein_type="collagen",
        mode=args.mode,
        region=(region[0] - 1, region[1]) if region else None,
        top_n=args.top_n,
        max_sites=2,
        beam_width=6,
        structure_stats=structure_stats,
    )
    result = run_design(request)
    elapsed = time.time() - started

    print(f" 扫描位点   : {result.plan['position_count']}（候选 {result.plan['candidate_count']} 条）")
    print(f" 被保护位点 : {len(result.plan['protected'])}")
    print(f" 耗时       : {elapsed:.1f}s")

    report["design"] = {
        "mode": result.mode,
        "position_count": result.plan["position_count"],
        "candidate_count": result.plan["candidate_count"],
        "protected_count": len(result.plan["protected"]),
        "exclusion_count": len(result.plan["exclusions"]),
        "downsampled": result.plan["downsampled"],
        "elapsed_seconds": round(elapsed, 2),
        "summary": {k: v for k, v in result.summary.items() if k != "hotspots"},
    }

    print("\n " + "-" * 74)
    print(" Top 候选（单点）")
    print(" " + "-" * 74)
    print(f" {'突变':<12}{'总分':>7}  {'稳定性':>6}{'活性':>6}{'折叠':>6}{'表达':>6}{'安全':>6}   告警")
    for candidate in result.candidates[:12]:
        scores = {dimension.key: dimension.raw for dimension in candidate.dimensions}
        flags = "；".join(candidate.flags)[:34] or "—"
        print(
            f" {candidate.mutations[0]:<12}{candidate.total_score:>7.2f}  "
            f"{scores.get('stability', 0):>6.0f}{scores.get('activity', 0):>6.0f}"
            f"{scores.get('foldability', 0):>6.0f}{scores.get('expression', 0):>6.0f}"
            f"{scores.get('risk', 0):>6.0f}   {flags}"
        )
    report["top_candidates"] = [candidate.to_dict() for candidate in result.candidates[:args.top_n]]

    # ---------- 5) 胶原领域知识一致性校验 ----------
    # 期望：所有推荐位点都不落在三股螺旋 Gly 位上（那是绝对禁区）
    violations: list[str] = []
    for candidate in result.candidates:
        for position in candidate.positions:
            if position in protected:
                violations.append(f"{candidate.mutations[0]} 触及受保护位点 {position + 1}")

    y_site_proposals = 0
    x_site_proposals = 0
    for candidate in result.candidates:
        for position in candidate.positions:
            phase = gly_xy_phase(sequence, position)
            if phase == 2 and candidate.mutations[0].endswith("P"):
                y_site_proposals += 1
            elif phase == 1:
                x_site_proposals += 1

    print("\n " + "-" * 74)
    print(" 领域知识一致性校验")
    print(" " + "-" * 74)
    print(f"  触及受保护位点的候选        : {len(violations)}  （期望 0）")
    print(f"  Y 位脯氨酸（羟化位点）推荐数 : {y_site_proposals}")
    print(f"  X 位改造推荐数              : {x_site_proposals}")

    report["consistency_check"] = {
        "protected_violations": violations,
        "protected_violation_count": len(violations),
        "y_site_proline_proposals": y_site_proposals,
        "x_site_proposals": x_site_proposals,
        "passed": len(violations) == 0,
    }

    print("\n " + "-" * 74)
    print(" 改造策略建议")
    print(" " + "-" * 74)
    for hint in result.hints:
        print(f"  · {hint}")
    report["hints"] = result.hints

    if result.warnings:
        print("\n " + "-" * 74)
        print(" 提示与限制")
        print(" " + "-" * 74)
        for warning in result.warnings:
            print(f"  ⚠ {warning}")
    report["warnings"] = result.warnings

    # ---------- 6) 落盘 ----------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "case_collagen.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n 结果已写入: {output}")
    print("=" * 78)

    return 0 if report["consistency_check"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
