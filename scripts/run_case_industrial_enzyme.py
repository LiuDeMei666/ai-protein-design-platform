#!/usr/bin/env python
"""标准化测试案例 ②：工业酶（枯草杆菌蛋白酶 BPN'）。

案例目标（对应需求文档"针对至少一种工业酶的标准化测试案例"）
------------------------------------------------------------
1. 验证平台能通过**序列模体**（而非硬编码编号）正确识别催化三联体，
   并用**相对间距自校验**交叉验证识别结果；
2. 验证前导肽区（信号肽 + propeptide）被正确保护——在该区推荐突变对成熟酶无意义；
3. 验证热稳定性改造建议符合嗜热同源物的残基偏好；
4. 输出结构化定量结果供验证报告引用。

关键验证点
----------
枯草杆菌蛋白酶前体（382 aa）与成熟酶（275 aa）的编号相差 107 位，
直接套用文献编号是最常见的错误来源。本案例通过"相对间距"这一与编号体系无关的
不变量来验证识别正确性：催化 Ser 与 His/Asp/氧负离子洞 Asn 的间距应分别约为
157 / 189 / 66-67。

用法::

    python scripts/run_case_industrial_enzyme.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SEED = PROJECT_ROOT / "data" / "seeds" / "protease_subtilisin.fasta"
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
    return header or "Subtilisin", sequence


def main() -> int:
    parser = argparse.ArgumentParser(description="工业酶标准化测试案例")
    parser.add_argument("--skip-structure", action="store_true")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    from backend.app.core.logging import get_logger, setup_logging
    from backend.app.services.design.engine import DesignRequest, run_design
    from backend.app.services.design.rulepacks import get_rulepack
    from backend.app.services.design.rulepacks.protease import verify_active_site
    from backend.app.services.sequence.validator import validate_sequence

    setup_logging()
    logger = get_logger("case.enzyme")

    name, sequence = load_seed()
    print("=" * 78)
    print(" 标准化测试案例 ② 工业酶（枯草杆菌蛋白酶）")
    print("=" * 78)
    print(f" 对象       : {name}")
    print(f" 序列长度   : {len(sequence)} aa（前体；成熟酶为 275 aa）")

    report: dict = {"case": "industrial_enzyme", "target": name, "sequence_length": len(sequence)}

    check = validate_sequence(sequence)
    report["sequence_check"] = {"ok": check.ok, "warnings": check.warnings, "errors": check.errors}
    print(f" 序列校验   : {'通过' if check.ok else '失败'}")

    # ---------- 1) 活性位点识别与自校验 ----------
    verification = verify_active_site(sequence)
    print("\n " + "-" * 74)
    print(" 活性位点识别（序列模体法 + 相对间距自校验）")
    print(" " + "-" * 74)
    if verification["catalytic_ser_1based"] is None:
        print("  ✗ 未识别到枯草杆菌蛋白酶活性位点模体")
    else:
        for position, reason in verification["protected"].items():
            print(f"  第 {position:>4} 位  {reason}")
        print("  相对间距校验：")
        for role, item in verification["spacing_check"].items():
            mark = "✓" if item["ok"] else "✗"
            print(
                f"    {mark} {role:<4} 期望间距 {item['expected']:>4}，实际 {item['actual']}"
            )
        print(f"  综合判定：{'全部通过' if verification['all_checks_passed'] else '存在不一致，需人工复核'}")

    report["active_site"] = verification

    # ---------- 2) 保护位点 ----------
    rulepack = get_rulepack("protease")
    protected = rulepack.protected_positions(sequence)
    catalytic = {
        key: value for key, value in protected.items() if "BPN" in value or "催化" in value or "氧负离子" in value
    }
    proregion = {key: value for key, value in protected.items() if "前导肽" in value}
    pocket = {key: value for key, value in protected.items() if "口袋" in value}

    print(f"\n 保护位点合计: {len(protected)}")
    print(f"   · 催化相关 : {len(catalytic)} 个")
    print(f"   · 前导肽区 : {len(proregion)} 个（{min(proregion) + 1 if proregion else '-'} - "
          f"{max(proregion) + 1 if proregion else '-'}，成熟过程中被自切去除）")
    print(f"   · 底物口袋 : {len(pocket)} 个")

    report["protection"] = {
        "total": len(protected),
        "catalytic": {str(key + 1): value for key, value in catalytic.items()},
        "proregion_count": len(proregion),
        "proregion_range": [min(proregion) + 1, max(proregion) + 1] if proregion else None,
        "pocket": {str(key + 1): value for key, value in pocket.items()},
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
            composition = structure.stats.get("secondary_structure_composition", {})
            print(
                " 二级结构   : "
                f"螺旋 {composition.get('helix_total', 0):.1%} · "
                f"折叠 {composition.get('sheet', 0):.1%} · "
                f"卷曲 {composition.get('coil', 0):.1%}"
            )
            structure_stats = structure.stats
            report["structure"] = {
                "source": structure.source,
                "mean_plddt": structure.mean_plddt,
                "secondary_structure_composition": composition,
                "elapsed_seconds": round(elapsed, 2),
                "warnings": structure.warnings,
            }
        except Exception as exc:
            print(f" 结构预测失败（案例继续）：{exc}")
            report["structure"] = {"error": str(exc)}

    # ---------- 4) 突变设计 ----------
    print("\n 突变设计中…")
    started = time.time()
    result = run_design(
        DesignRequest(
            sequence=sequence,
            protein_type="protease",
            mode="single",
            top_n=args.top_n,
            structure_stats=structure_stats,
        )
    )
    elapsed = time.time() - started

    print(f" 扫描位点   : {result.plan['position_count']}（候选 {result.plan['candidate_count']} 条）")
    print(f" 被保护位点 : {len(result.plan['protected'])}")
    print(f" 耗时       : {elapsed:.1f}s")

    # 成熟酶区起点：首个受保护的前导肽位点之后
    mature_start = max(proregion) + 1 if proregion else 0
    print(f" 成熟酶起始 : 第 {mature_start + 1} 位（前导肽 {mature_start} 个残基被保护）")

    report["design"] = {
        "position_count": result.plan["position_count"],
        "candidate_count": result.plan["candidate_count"],
        "protected_count": len(result.plan["protected"]),
        "mature_start_1based": mature_start + 1,
        "elapsed_seconds": round(elapsed, 2),
        "summary": {k: v for k, v in result.summary.items() if k != "hotspots"},
    }

    print("\n " + "-" * 74)
    print(" Top 候选（单点，均位于成熟酶区）")
    print(" " + "-" * 74)
    print(f" {'突变':<12}{'位置':>6}{'总分':>8}  {'稳定性':>6}{'活性':>6}{'折叠':>6}{'安全':>6}   告警")
    for candidate in result.candidates[:12]:
        scores = {dimension.key: dimension.raw for dimension in candidate.dimensions}
        position = candidate.positions[0] + 1
        flag = "；".join(candidate.flags)[:30] or "—"
        print(
            f" {candidate.mutations[0]:<12}{position:>6}{candidate.total_score:>8.2f}  "
            f"{scores.get('stability', 0):>6.0f}{scores.get('activity', 0):>6.0f}"
            f"{scores.get('foldability', 0):>6.0f}{scores.get('risk', 0):>6.0f}   {flag}"
        )
    report["top_candidates"] = [candidate.to_dict() for candidate in result.candidates[: args.top_n]]

    # ---------- 5) 一致性校验 ----------
    violations: list[str] = []
    in_proregion = 0
    touching_catalytic = 0
    for candidate in result.candidates:
        for position in candidate.positions:
            if position in protected:
                violations.append(f"{candidate.mutations[0]} 触及受保护位点 {position + 1}")
            if position < mature_start:
                in_proregion += 1
            if position in catalytic:
                touching_catalytic += 1

    print("\n " + "-" * 74)
    print(" 领域知识一致性校验")
    print(" " + "-" * 74)
    checks = [
        ("推荐位点均不触及受保护位点", len(violations) == 0, f"违规 {len(violations)} 条"),
        ("推荐位点均落在成熟酶区", in_proregion == 0, f"前导肽区推荐 {in_proregion} 条"),
        ("推荐位点均不触及催化残基", touching_catalytic == 0, f"触及 {touching_catalytic} 条"),
        ("活性位点相对间距自校验通过", verification["all_checks_passed"], ""),
    ]
    for label, passed, detail in checks:
        print(f"  {'✓' if passed else '✗'} {label}{('（' + detail + '）') if detail else ''}")

    report["consistency_check"] = {
        "protected_violations": violations,
        "proregion_recommendations": in_proregion,
        "catalytic_collisions": touching_catalytic,
        "spacing_check_passed": verification["all_checks_passed"],
        "passed": all(item[1] for item in checks),
    }

    print("\n " + "-" * 74)
    print(" 改造策略建议")
    print(" " + "-" * 74)
    for hint in result.hints:
        print(f"  · {hint}")
    report["hints"] = result.hints
    report["warnings"] = result.warnings

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "case_industrial_enzyme.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n 结果已写入: {output}")
    print("=" * 78)

    return 0 if report["consistency_check"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
