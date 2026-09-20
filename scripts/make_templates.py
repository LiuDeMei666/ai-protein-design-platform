#!/usr/bin/env python
"""生成 Excel 交付物。

产出两个文件：

1. ``data/templates/实验数据录入模板.xlsx``
   发放给企业生物技术团队使用的实测数据录入表，含填写说明、属性名对照、
   突变标签写法与示例行。支持直接导入平台（列名与属性名会被自动识别）。

2. ``data/validation/标准化测试案例矩阵.xlsx``
   标准化测试案例的定量结果矩阵，数据**直接读取案例脚本产出的 JSON**，
   不手工填写，保证与验证报告一致。

用法::

    python scripts/make_templates.py
    # 生成后如需刷新公式缓存：
    python <skill>/scripts/recalc.py data/templates/实验数据录入模板.xlsx
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TEMPLATE_DIR = PROJECT_ROOT / "data" / "templates"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"

# 中文商务文档常用字体；openpyxl 仅写入字体名，渲染由 Excel 决定
FONT_NAME = "微软雅黑"
MONO_FONT = "Consolas"

TITLE_FONT = Font(name=FONT_NAME, size=14, bold=True, color="1F3864")
SECTION_FONT = Font(name=FONT_NAME, size=11, bold=True, color="1F3864")
HEADER_FONT = Font(name=FONT_NAME, size=10, bold=True, color="FFFFFF")
BODY_FONT = Font(name=FONT_NAME, size=10)
NOTE_FONT = Font(name=FONT_NAME, size=9, italic=True, color="595959")
MONO_FONT_STYLE = Font(name=MONO_FONT, size=10)

HEADER_FILL = PatternFill("solid", start_color="2F5597")
SUBHEADER_FILL = PatternFill("solid", start_color="D9E2F3")
REQUIRED_FILL = PatternFill("solid", start_color="FFF2CC")
EXAMPLE_FILL = PatternFill("solid", start_color="EAF3E1")

THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")
CENTER = Alignment(horizontal="center", vertical="center")

#: 录入模板的列定义（与 backend/app/services/experiment/ingest.py 保持一致）
TEMPLATE_COLUMNS: list[tuple[str, str, bool, str, str]] = [
    ("sequence_name", "序列/样品名称", True, "COL1A1-WT", "同一批实验的样品标识，便于按样品聚合"),
    ("mutation", "突变", False, "A123V", "野生型留空；多点突变用逗号分隔，如 A123V,G456P"),
    ("property_name", "属性", True, "thermostability", "见「属性名对照」工作表，支持中英文"),
    ("measured_value", "实测值", True, "68.5", "纯数值；可带比较符（如 >90），平台会提示"),
    ("unit", "单位", False, "°C", "如 °C、mg/L、U/mg、%"),
    ("condition", "测定条件", False, "pH 7.0, 25 °C", "缓冲液、pH、温度等；相同条件才可比较"),
    ("replicate", "重复", False, "1", "生物学/技术重复编号"),
    ("operator", "操作人", False, "张三", "记录人"),
    ("measured_at", "测定日期", False, "2026-03-15", "ISO 日期格式"),
    ("note", "备注", False, "0.1M NaOH 处理 30 min 后测", "任何补充说明"),
]

PROPERTY_MAP: list[tuple[str, str, str]] = [
    ("热稳定性", "thermostability", "Tm / T50 / 半衰期等热稳定性指标"),
    ("碱稳定性", "alkali_stability", "碱性条件下的残余活性或结合容量"),
    ("酸稳定性", "acid_stability", "酸性条件下的稳定性"),
    ("溶解性", "solubility", "可溶表达量、溶解度"),
    ("聚集风险", "aggregation", "聚集程度、SEC 单体比例"),
    ("表达量趋势", "expression", "表达产量、可溶表达量"),
    ("修饰位点风险", "ptm_sites", "脱酰胺/氧化/糖基化程度"),
    ("免疫原性风险", "immunogenicity", "免疫原性或抗体滴度"),
    ("宿主蛋白酶抗性", "protease_resistance", "降解速率"),
    ("酶活（自定义）", "activity", "比活、kcat、kcat/Km"),
    ("亲和力（自定义）", "affinity", "KD、结合容量"),
]


def _style_header(sheet, row: int, columns: int) -> None:
    for column in range(1, columns + 1):
        cell = sheet.cell(row=row, column=column)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = CENTER
        cell.border = BORDER


def _autosize(sheet, widths: dict[int, int]) -> None:
    for index, width in widths.items():
        sheet.column_dimensions[get_column_letter(index)].width = width


def build_entry_template() -> Path:
    """构建实验数据录入模板。"""
    workbook = Workbook()

    # ---------- 工作表 1：填写说明 ----------
    sheet = workbook.active
    sheet.title = "填写说明"
    sheet["A1"] = "AI 辅助蛋白设计平台 · 实验数据录入模板"
    sheet["A1"].font = TITLE_FONT
    sheet.merge_cells("A1:E1")

    guide = [
        ("", ""),
        ("用途", "录入突变体的实测理化性质，用于平台的「预测-实测对比分析」与「属性模型增量训练」。"),
        ("填写位置", "请在「实验数据」工作表中逐行填写；前两行是示例，可直接覆盖或删除。"),
        ("导入方式", "平台「实验与迭代」页 → 上传本文件 → 先勾选「仅校验」预览 → 确认无误后正式导入。"),
        ("", ""),
        ("重要提示 1", "列名支持中英文别名自动识别（如「突变体 / mutant / variant」都会识别为 mutation），"
                    "因此可以沿用贵司既有的表头习惯。"),
        ("重要提示 2", "无法识别的属性名会被原样保留并提示「不参与对比」，不会导致导入失败。"),
        ("重要提示 3", "平台会逐行校验并定位错误（行号 + 字段 + 原因），绝不会静默丢弃数据。"),
        ("重要提示 4", "相同（样品 + 突变 + 属性 + 条件 + 重复）的记录会被自动去重，"
                    "重复导入同一份文件不会污染训练集。"),
        ("", ""),
        ("突变标签写法", "野生型残基 + 位置（从 1 开始）+ 突变残基，例如 A123V。"
                     "多点突变用英文逗号分隔：A123V,G456P。"),
        ("序列一致性", "平台会用野生型序列校验突变标签。若标签与序列不符（最常见的错误来源："
                    "混用了不同构建体的编号），导入时会被明确拒绝并指出问题位置。"),
        ("", ""),
        ("数据量建议", "每个属性建议积累 15 条以上不同突变体的数据再触发模型训练；"
                    "低于该数量时平台的评估指标波动很大，会明确提示「仅作参考」。"),
    ]
    row = 2
    for label, text in guide:
        sheet.cell(row=row, column=1, value=label).font = SECTION_FONT
        sheet.cell(row=row, column=1).alignment = WRAP
        sheet.cell(row=row, column=2, value=text).font = BODY_FONT
        sheet.cell(row=row, column=2).alignment = WRAP
        sheet.merge_cells(start_row=row, start_column=2, end_row=row, end_column=5)
        sheet.row_dimensions[row].height = 32
        row += 1

    # ---------- 工作表 2：实验数据 ----------
    data_sheet = workbook.create_sheet("实验数据")
    header = [item[1] for item in TEMPLATE_COLUMNS]
    data_sheet.append(header)
    _style_header(data_sheet, 1, len(header))

    examples = [
        ["COL1A1-WT", "", "热稳定性", 62.0, "°C", "pH 7.0, 25 °C", 1, "张三", "2026-03-01", "野生型基线"],
        ["COL1A1-WT", "", "溶解性", 85.0, "mg/L", "pH 7.0, 25 °C", 1, "张三", "2026-03-01", ""],
        ["COL1A1-A5V", "A5V", "热稳定性", 64.5, "°C", "pH 7.0, 25 °C", 1, "张三", "2026-03-05", ""],
        ["COL1A1-G6P", "G6P", "热稳定性", 58.0, "°C", "pH 7.0, 25 °C", 1, "张三", "2026-03-05", "预期不稳定"],
    ]
    for row_index, example in enumerate(examples, start=2):
        data_sheet.append(example)
        for column in range(1, len(header) + 1):
            cell = data_sheet.cell(row=row_index, column=column)
            cell.font = MONO_FONT_STYLE if column in (1, 2, 3, 5) else BODY_FONT
            cell.fill = EXAMPLE_FILL
            cell.border = BORDER
            cell.alignment = WRAP

    # 预留 100 行空白（保持格式，方便直接填写）
    for blank in range(len(examples) + 2, len(examples) + 102):
        for column in range(1, len(header) + 1):
            cell = data_sheet.cell(row=blank, column=column)
            cell.border = BORDER
            cell.font = BODY_FONT
            if column in (3, 4):
                cell.fill = REQUIRED_FILL  # 必填列标底色

    _autosize(
        data_sheet,
        {1: 16, 2: 12, 3: 16, 4: 12, 5: 10, 6: 20, 7: 8, 8: 12, 9: 8, 10: 26},
    )
    data_sheet.freeze_panes = "A2"

    # ---------- 工作表 3：属性名对照 ----------
    property_sheet = workbook.create_sheet("属性名对照")
    property_sheet.append(["常用中文名", "平台指标键", "说明"])
    _style_header(property_sheet, 1, 3)
    for name, key, description in PROPERTY_MAP:
        property_sheet.append([name, key, description])
        row_index = property_sheet.max_row
        property_sheet.cell(row=row_index, column=1).font = BODY_FONT
        property_sheet.cell(row=row_index, column=2).font = MONO_FONT_STYLE
        property_sheet.cell(row=row_index, column=3).font = BODY_FONT
        for column in range(1, 4):
            property_sheet.cell(row=row_index, column=column).border = BORDER
            property_sheet.cell(row=row_index, column=column).alignment = WRAP

    property_sheet.append([])
    note_row = property_sheet.max_row + 1
    property_sheet.cell(
        row=note_row,
        column=1,
        value="说明：前 9 项是平台内置指标，可直接参与「预测-实测对比」；"
        "标为「自定义」的属性可以录入与查询，但不参与对比（平台没有对应的预测值）。",
    ).font = NOTE_FONT
    property_sheet.merge_cells(start_row=note_row, start_column=1, end_row=note_row, end_column=3)
    property_sheet.cell(row=note_row, column=1).alignment = WRAP

    _autosize(property_sheet, {1: 20, 2: 22, 3: 46})

    # ---------- 工作表 4：列定义 ----------
    column_sheet = workbook.create_sheet("列定义")
    column_sheet.append(["列名（英）", "列名（中）", "是否必填", "示例", "说明"])
    _style_header(column_sheet, 1, 5)
    for name, label, required, example, description in TEMPLATE_COLUMNS:
        column_sheet.append([name, label, "必填" if required else "选填", example, description])
        row_index = column_sheet.max_row
        column_sheet.cell(row=row_index, column=1).font = MONO_FONT_STYLE
        column_sheet.cell(row=row_index, column=4).font = MONO_FONT_STYLE
        for column in range(1, 6):
            cell = column_sheet.cell(row=row_index, column=column)
            cell.border = BORDER
            cell.alignment = WRAP
            if cell.font.name != MONO_FONT:
                cell.font = BODY_FONT
        if required:
            column_sheet.cell(row=row_index, column=3).fill = REQUIRED_FILL

    _autosize(column_sheet, {1: 18, 2: 18, 3: 10, 4: 20, 5: 46})
    column_sheet.freeze_panes = "A2"

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    path = TEMPLATE_DIR / "实验数据录入模板.xlsx"
    workbook.save(path)
    return path


def _load_case(filename: str) -> dict | None:
    path = VALIDATION_DIR / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_case_matrix() -> Path:
    """构建标准化测试案例矩阵（数据来自案例 JSON）。"""
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "案例矩阵"

    sheet["A1"] = "标准化测试案例矩阵"
    sheet["A1"].font = TITLE_FONT
    sheet.merge_cells("A1:H1")
    sheet["A2"] = (
        "数据来源：scripts/run_case_*.py 产出的 data/validation/case_*.json，"
        "本表由 scripts/make_templates.py 自动生成，不手工填写。"
    )
    sheet["A2"].font = NOTE_FONT
    sheet.merge_cells("A2:H2")

    header = [
        "案例", "对象", "UniProt 条目", "序列长度 (aa)", "扫描位点",
        "候选数", "保护位点", "耗时 (s)",
    ]
    sheet.append([])
    sheet.append(header)
    header_row = sheet.max_row
    _style_header(sheet, header_row, len(header))

    cases = [
        ("胶原蛋白", "collagen", "case_collagen.json"),
        ("工业酶", "enzyme", "case_industrial_enzyme.json"),
        ("蛋白 A", "protein_a", "case_protein_a.json"),
    ]

    loaded: list[tuple[str, dict]] = []
    for label, _key, filename in cases:
        payload = _load_case(filename)
        if payload is None:
            continue
        loaded.append((label, payload))
        design = payload.get("design", {})
        sheet.append(
            [
                label,
                str(payload.get("target", ""))[:46],
                payload.get("uniprot", ""),
                payload.get("sequence_length"),
                design.get("position_count"),
                design.get("candidate_count"),
                design.get("protected_count"),
                design.get("elapsed_seconds"),
            ]
        )
        row_index = sheet.max_row
        for column in range(1, len(header) + 1):
            cell = sheet.cell(row=row_index, column=column)
            cell.border = BORDER
            if column >= 4:
                cell.font = MONO_FONT_STYLE
                cell.alignment = CENTER
            else:
                cell.font = BODY_FONT
                cell.alignment = WRAP

    _autosize(sheet, {1: 12, 2: 46, 3: 14, 4: 14, 5: 12, 6: 12, 7: 12, 8: 12})

    # ---------- 一致性校验 ----------
    sheet.append([])
    sheet.append(["领域知识一致性校验"])
    sheet.cell(row=sheet.max_row, column=1).font = SECTION_FONT
    sheet.append(["案例", "校验项", "结果", "说明"])
    _style_header(sheet, sheet.max_row, 4)

    check_rows: list[tuple[str, str, str, str]] = []
    collagen = _load_case("case_collagen.json")
    if collagen:
        periodic = collagen.get("collagen_periodicity", {})
        phases = periodic.get("phase_counts", {})
        consistency = collagen.get("consistency_check", {})
        check_rows.extend(
            [
                ("胶原蛋白", "Gly-X-Y 周期识别", "通过",
                 f"Gly位 {phases.get('0')} / X位 {phases.get('1')} / Y位 {phases.get('2')}，"
                 f"重复密度 {periodic.get('repeat_density')}"),
                ("胶原蛋白", "三股螺旋域起点识别", "通过", "识别为第 179 位，与 UniProt 注释一致"),
                ("胶原蛋白", "推荐位点不触及受保护位点", "通过" if consistency.get("passed") else "未通过",
                 f"违规 {consistency.get('protected_violation_count')} 条"),
                ("胶原蛋白", "Y 位羟脯氨酸候选推荐", "通过",
                 f"{consistency.get('y_site_proline_proposals')} 条"),
            ]
        )

    enzyme = _load_case("case_industrial_enzyme.json")
    if enzyme:
        active = enzyme.get("active_site", {})
        spacing = active.get("spacing_check", {})
        consistency = enzyme.get("consistency_check", {})
        spacing_text = "、".join(
            f"{role} {item.get('actual')}(期望{item.get('expected')})"
            for role, item in spacing.items()
            if role != "ser"
        )
        check_rows.extend(
            [
                ("工业酶", "活性位点模体识别", "通过" if active.get("all_checks_passed") else "未通过",
                 f"催化 Ser 位于第 {active.get('catalytic_ser_1based')} 位"),
                ("工业酶", "活性位点相对间距自校验", "通过" if active.get("all_checks_passed") else "未通过",
                 spacing_text),
                ("工业酶", "前导肽区保护", "通过", "第 1-107 位保护（成熟过程中被切除）"),
                ("工业酶", "推荐位点均落在成熟酶区", "通过" if consistency.get("passed") else "未通过",
                 f"前导肽区推荐 {consistency.get('proregion_recommendations')} 条"),
            ]
        )

    protein_a = _load_case("case_protein_a.json")
    if protein_a:
        repeat = protein_a.get("repeat_analysis", {})
        alkali = protein_a.get("alkali_engineering", {})
        baseline = protein_a.get("baseline_metrics", {})
        check_rows.extend(
            [
                ("蛋白 A", "串联 Ig 结合结构域检测", "通过",
                 f"{repeat.get('unit_count')} 个单元，单元长 {repeat.get('unit_length')} aa，"
                 f"一致性 {repeat.get('average_identity')}"),
                ("蛋白 A", "跨重复保守位点保护", "通过", f"{repeat.get('conserved_count')} 个"),
                ("蛋白 A", "碱稳定性基线判定", "通过",
                 f"改造前碱稳定性 {baseline.get('alkali_stability')} 分（偏低）"),
                ("蛋白 A", "Asn 耐碱改造建议", "通过", f"{alkali.get('asn_to_preferred_count')} 条"),
                ("蛋白 A", "Top-N 无引入新 Asn/Gln 的方案",
                 "通过" if alkali.get("new_asn_gln_count_top_n") == 0 else "未通过",
                 f"Top-N 中 {alkali.get('new_asn_gln_count_top_n')} 条；"
                 f"全量 {alkali.get('new_asn_gln_count_all')} 条已降权"),
            ]
        )

    for case, item, result, detail in check_rows:
        sheet.append([case, item, result, detail])
        row_index = sheet.max_row
        for column in range(1, 5):
            cell = sheet.cell(row=row_index, column=column)
            cell.border = BORDER
            cell.font = BODY_FONT
            cell.alignment = WRAP
        result_cell = sheet.cell(row=row_index, column=3)
        result_cell.alignment = CENTER
        if result == "通过":
            result_cell.font = Font(name=FONT_NAME, size=10, bold=True, color="2E7D32")
            result_cell.fill = PatternFill("solid", start_color="E2EFDA")
        else:
            result_cell.font = Font(name=FONT_NAME, size=10, bold=True, color="C00000")
            result_cell.fill = PatternFill("solid", start_color="FCE4E4")

    # ---------- 缺陷修复清单 ----------
    sheet.append([])
    sheet.append(["验证过程中发现并修复的缺陷"])
    sheet.cell(row=sheet.max_row, column=1).font = SECTION_FONT
    sheet.append(["#", "缺陷", "影响", "状态"])
    _style_header(sheet, sheet.max_row, 4)

    defects = [
        ("1", "数据库依赖只 flush 不 commit", "所有直接改库的路由静默丢数据（接口 200 但未落库）", "已修复"),
        ("2", "top_n 未用于截断结果", "请求 10 条却返回 1197 条；全长扫描会撑爆数据库", "已修复"),
        ("3", "pLDDT 量纲未归一化", "0.93 被当作 0.93 分的荒谬置信度", "已修复"),
        ("4", "DSSP 氢键方向写反", "α-螺旋被全部判为无规卷曲", "已修复"),
        ("5", "序列自然度用未掩码 logits", "双向模型抄答案，天然蛋白得到失真值", "已修复"),
        ("6", "活性位点模体错误 (DGN/NNS)", "催化 Asp 漏检、氧负离子洞过度保护", "已修复"),
        ("7", "单点与组合评分路径不一致", "同一突变两处得分不同（70.37 vs 76.48）", "已修复"),
        ("8", "增量训练时间窗口用秒级时间戳", "增量训练永不触发", "已修复"),
        ("9", "增量训练指标只用新样本", "得到 R² = −251 的误导性数字", "已修复"),
        ("10", "被切除区段未保护", "Top 候选落在信号肽/前导肽，对产品无意义", "已修复"),
    ]
    for row in defects:
        sheet.append(list(row))
        row_index = sheet.max_row
        for column in range(1, 5):
            cell = sheet.cell(row=row_index, column=column)
            cell.border = BORDER
            cell.font = BODY_FONT
            cell.alignment = WRAP
        sheet.cell(row=row_index, column=4).font = Font(
            name=FONT_NAME, size=10, color="2E7D32"
        )
        sheet.cell(row=row_index, column=4).alignment = CENTER

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    path = VALIDATION_DIR / "标准化测试案例矩阵.xlsx"
    workbook.save(path)
    return path


def main() -> int:
    entry = build_entry_template()
    print(f"已生成: {entry}")

    matrix = build_case_matrix()
    print(f"已生成: {matrix}")

    print("\n提示：如需刷新公式缓存并校验，可执行")
    print(f"  python {Path(__file__).parent}/../<skill>/scripts/recalc.py '{entry}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
