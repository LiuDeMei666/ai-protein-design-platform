#!/usr/bin/env python
"""标准化测试案例 ③：蛋白 A（金黄色葡萄球菌 Protein A / spa）。

案例目标
--------
1. 验证平台能自动检测**串联 Ig 结合结构域**（蛋白 A 由 5 个高度同源结构域组成），
   并通过"跨重复保守位点"分析保护框架与 Fc 结合界面——这比硬编码残基编号稳健得多；
2. 验证耐碱性改造建议符合业界成熟策略（消除 Asn 脱酰胺位点，尤其 Asn-Gly）；
3. 验证不会推荐引入新的 Asn/Gln（那会新增脱酰胺风险，与耐碱目标冲突）；
4. 输出结构化定量结果供验证报告引用。

背景知识
--------
IgG 亲和层析的 0.1 M NaOH 清洗步骤是蛋白 A 失活的主因，而失活主要来自
**Asn 脱酰胺**（尤其 Asn-Gly 基序）。因此"提升耐碱性"的核心就是把关键 Asn
替换为 Gln/Asp/Thr/Ser/Ala。本案例会统计平台给出的此类建议数量。

用法::

    python scripts/run_case_protein_a.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SEED = PROJECT_ROOT / "data" / "seeds" / "protein_a_staph.fasta"
OUTPUT_DIR = PROJECT_ROOT / "data" / "validation"


def load_seed() -> tuple[str, str]:
    if not SEED.exists():
        raise SystemExit(f"缺少种子序列 {SEED}\n请先执行: python scripts/fetch_seed_sequences.py")
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
    return header or "ProteinA", sequence


def main() -> int:
    parser = argparse.ArgumentParser(description="蛋白 A 标准化测试案例")
    parser.add_argument("--skip-structure", action="store_true")
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args()

    from backend.app.core.logging import setup_logging
    from backend.app.services.design.engine import DesignRequest, run_design
    from backend.app.services.design.rulepacks import get_rulepack
    from backend.app.services.design.rulepacks.protein_a import (
        conserved_positions_across_repeats,
        detect_tandem_repeats,
    )
    from backend.app.services.property.metric import METRIC_LABELS
    from backend.app.services.property.pipeline import predict_properties
    from backend.app.services.sequence.validator import validate_sequence

    setup_logging()

    name, sequence = load_seed()
    print("=" * 78)
    print(" 标准化测试案例 ③ 蛋白 A")
    print("=" * 78)
    print(f" 对象       : {name}")
    print(f" 序列长度   : {len(sequence)} aa")

    report: dict = {"case": "protein_a", "target": name, "sequence_length": len(sequence)}

    check = validate_sequence(sequence)
    report["sequence_check"] = {"ok": check.ok, "warnings": check.warnings, "errors": check.errors}
    print(f" 序列校验   : {'通过' if check.ok else '失败'}")

    # ---------- 1) 串联重复结构域检测 ----------
    units, identity = detect_tandem_repeats(sequence)
    conserved = conserved_positions_across_repeats(sequence, units)

    print("\n " + "-" * 74)
    print(" 串联 Ig 结合结构域检测")
    print(" " + "-" * 74)
    if units:
        print(f"  检测到 {len(units)} 个重复单元，长度约 {units[0].length} aa，"
              f"单元间平均一致性 {identity:.1%}")
        for index, unit in enumerate(units, start=1):
            print(f"    单元 {index}: 第 {unit.start + 1}-{unit.end} 位")
        print(f"  跨重复保守位点: {len(conserved)} 个（框架 + Fc 结合界面）")
    else:
        print("  ✗ 未检测到串联重复结构域")

    report["repeat_analysis"] = {
        "unit_count": len(units),
        "unit_length": units[0].length if units else 0,
        "average_identity": identity,
        "unit_ranges": [[unit.start + 1, unit.end] for unit in units],
        "conserved_count": len(conserved),
    }

    # ---------- 2) 脱酰胺风险基线（改造前） ----------
    alkali_report = predict_properties(
        sequence, protein_type="protein_a", use_esm=False, structure_stats=None
    )
    baseline = {key: metric.score for key, metric in alkali_report.metrics.items()}
    print("\n " + "-" * 74)
    print(" 改造前性质基线")
    print(" " + "-" * 74)
    for key in ("thermostability", "alkali_stability", "acid_stability", "solubility", "aggregation"):
        if key in baseline:
            print(f"  {METRIC_LABELS.get(key, key):<14}: {baseline[key]:>5.1f}")
    report["baseline_metrics"] = baseline

    asn_gly = sum(1 for index in range(len(sequence) - 1) if sequence[index : index + 2] == "NG")
    asn_total = sequence.count("N")
    gln_total = sequence.count("Q")
    print(f"  Asn-Gly 高敏感脱酰胺基序: {asn_gly} 个")
    print(f"  Asn 总数: {asn_total} · Gln 总数: {gln_total}")
    report["deamidation_baseline"] = {
        "asn_gly_motifs": asn_gly,
        "asn_count": asn_total,
        "gln_count": gln_total,
    }

    # ---------- 3) 结构预测 ----------
    structure_stats: dict | None = None
    if not args.skip_structure:
        from backend.app.services.structure.registry import predict_structure

        print("\n 结构预测中…")
        started = time.time()
        try:
            structure = predict_structure(sequence)
            elapsed = time.time() - started
            print(f" 结构来源   : {structure.source}  耗时 {elapsed:.1f}s")
            print(f" 平均 pLDDT : {structure.mean_plddt:.2f}")
            if structure.segments and len(structure.segments) > 1:
                print(f" 分片数     : {len(structure.segments)}（跨片段取向未建模）")
            structure_stats = structure.stats
            report["structure"] = {
                "source": structure.source,
                "mean_plddt": structure.mean_plddt,
                "fragment_count": len(structure.segments),
                "warnings": structure.warnings,
                "elapsed_seconds": round(elapsed, 2),
            }
        except Exception as exc:
            print(f" 结构预测失败（案例继续）：{exc}")
            report["structure"] = {"error": str(exc)}

    # ---------- 4) 突变设计（耐碱性方向） ----------
    print("\n 突变设计中（蛋白 A 规则包：优先消除脱酰胺位点）…")
    started = time.time()
    result = run_design(
        DesignRequest(
            sequence=sequence,
            protein_type="protein_a",
            mode="single",
            top_n=args.top_n,
            structure_stats=structure_stats,
        )
    )
    elapsed = time.time() - started

    print(f" 扫描位点   : {result.plan['position_count']}（候选 {result.plan['candidate_count']} 条）")
    print(f" 受保护位点 : {len(result.plan['protected'])}（跨重复保守位点）")
    print(f" 耗时       : {elapsed:.1f}s")

    report["design"] = {
        "position_count": result.plan["position_count"],
        "candidate_count": result.plan["candidate_count"],
        "protected_count": len(result.plan["protected"]),
        "elapsed_seconds": round(elapsed, 2),
        "summary": {k: v for k, v in result.summary.items() if k != "hotspots"},
    }

    # ---------- 5) 耐碱改造建议统计 ----------
    alkali_preferred = {"Q", "D", "E", "T", "S", "A"}
    asn_to_preferred = 0
    new_asn_gln = 0
    conserved_collision = 0

    for candidate in result.candidates:
        label = candidate.mutations[0]
        wild_type = label[0]
        mutant = label[-1]
        if wild_type == "N" and mutant in alkali_preferred:
            asn_to_preferred += 1
        if mutant in ("N", "Q") and wild_type not in ("N", "Q"):
            new_asn_gln += 1
        for position in candidate.positions:
            if position in conserved:
                conserved_collision += 1

    # 关键：规则包对"引入新 Asn/Gln"施加的是**降权**而不是剔除，
    # 因此这类候选仍会存在于全量候选中（只是排名靠后）。真正需要验证的是
    # **它们不应出现在推荐列表（Top-N）里**——这才能证明惩罚确实生效。
    top_n = result.candidates[: args.top_n]
    top_new_asn_gln = [
        candidate.mutations[0]
        for candidate in top_n
        if candidate.mutations[0][-1] in ("N", "Q") and candidate.mutations[0][0] not in ("N", "Q")
    ]
    print(f"\n  全量候选中引入新 Asn/Gln 的方案: {new_asn_gln} 条（已被降权，排名靠后）")
    print(f"  Top {len(top_n)} 推荐中引入新 Asn/Gln: {len(top_new_asn_gln)} 条"
          + (f" → {', '.join(top_new_asn_gln[:6])}" if top_new_asn_gln else ""))

    print("\n " + "-" * 74)
    print(" Top 耐碱改造候选（Asn -> Gln/Asp/Thr/Ser/Ala）")
    print(" " + "-" * 74)
    print(f" {'突变':<12}{'总分':>8}  {'稳定性':>6}{'活性':>6}{'安全':>6}   依据摘要")
    shown = 0
    for candidate in result.candidates:
        label = candidate.mutations[0]
        if label[0] == "N" and label[-1] in alkali_preferred:
            scores = {dimension.key: dimension.raw for dimension in candidate.dimensions}
            print(
                f" {label:<12}{candidate.total_score:>8.2f}  "
                f"{scores.get('stability', 0):>6.0f}{scores.get('activity', 0):>6.0f}"
                f"{scores.get('risk', 0):>6.0f}   {'；'.join(candidate.flags)[:36] or '—'}"
            )
            shown += 1
            if shown >= 10:
                break
    if shown == 0:
        print("  （Top 候选中未出现 Asn->Q/D/T/S/A 建议，可提高 top_n 查看）")

    print("\n " + "-" * 74)
    print(" 领域知识一致性校验")
    print(" " + "-" * 74)
    checks = [
        ("检测到串联 Ig 结合结构域", len(units) >= 2, f"{len(units)} 个单元"),
        ("保护位点来自跨重复保守分析", len(conserved) > 0, f"{len(conserved)} 个"),
        ("推荐不触及保守位点", conserved_collision == 0, f"违规 {conserved_collision} 条"),
        ("存在 Asn 耐碱改造建议", asn_to_preferred > 0, f"{asn_to_preferred} 条"),
        (
            f"Top {len(top_n)} 推荐中无引入新 Asn/Gln 的方案",
            len(top_new_asn_gln) == 0,
            f"Top-N 中 {len(top_new_asn_gln)} 条；全量中 {new_asn_gln} 条（已降权）",
        ),
    ]
    for label, passed, detail in checks:
        print(f"  {'✓' if passed else '✗'} {label}（{detail}）")

    report["alkali_engineering"] = {
        "asn_to_preferred_count": asn_to_preferred,
        "new_asn_gln_count_all": new_asn_gln,
        "new_asn_gln_count_top_n": len(top_new_asn_gln),
        "top_n_new_asn_gln_samples": top_new_asn_gln[:20],
        "conserved_collisions": conserved_collision,
    }
    report["consistency_check"] = {
        "checks": [{"label": label, "passed": passed, "detail": detail} for label, passed, detail in checks],
        "passed": all(item[1] for item in checks),
    }
    report["top_candidates"] = [candidate.to_dict() for candidate in result.candidates[: args.top_n]]

    print("\n " + "-" * 74)
    print(" 改造策略建议")
    print(" " + "-" * 74)
    for hint in result.hints:
        print(f"  · {hint}")
    report["hints"] = result.hints
    report["warnings"] = result.warnings

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "case_protein_a.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n 结果已写入: {output}")
    print("=" * 78)

    return 0 if report["consistency_check"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
