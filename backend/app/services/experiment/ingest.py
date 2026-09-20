"""实验数据结构化录入与校验。

设计目标
--------
企业生物技术团队日常用 Excel/CSV 记录实验结果，列名习惯各不相同（中英文混杂、
别名众多）。本模块负责：

1. **列名自动映射**：内置中英文别名表，把"突变体""实测值""检测条件"等
   自动映射到标准字段；映射结果回传前端供人工确认。
2. **逐行校验并明确报错**：不合法数值、缺失必填字段、属性名无法识别等问题
   都定位到具体行号与字段，**绝不静默丢弃数据**。
3. **属性名归一**：把"热稳定性/Tm/thermostability"统一到平台的指标键，
   保证后续"预测-实测对比"能正确配对。

不做的事：不修改用户原始数值（不做单位换算、不做异常值剔除），
只做标记与提示——化学/生物学上的判断应交由实验人员。
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ...core.logging import get_logger

logger = get_logger(__name__)

#: 标准字段 -> 可接受的列名别名（全部小写比较）
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "mutation": (
        "mutation", "mutant", "variant", "mut", "突变", "突变体", "变异", "突变位点", "氨基酸突变",
    ),
    "property_name": (
        "property_name", "property", "assay", "metric", "item", "属性", "性质", "指标",
        "检测项", "测定项目", "项目", "参数",
    ),
    "measured_value": (
        "measured_value", "value", "measurement", "result", "实测值", "测定值", "数值",
        "结果", "测量值", "实验值",
    ),
    "unit": ("unit", "units", "单位", "量纲"),
    "condition": (
        "condition", "conditions", "assay_condition", "条件", "检测条件", "测定条件",
        "实验条件", "测试条件",
    ),
    "replicate": ("replicate", "rep", "batch", "重复", "重复次", "批次", "序号"),
    "operator": ("operator", "person", "user", "操作人", "实验员", "负责人", "记录人"),
    "measured_at": (
        "measured_at", "date", "datetime", "日期", "测定日期", "实验日期", "检测日期", "时间",
    ),
    "note": ("note", "notes", "comment", "remark", "备注", "说明", "comments"),
    "sequence_name": ("sequence_name", "name", "protein", "序列名称", "蛋白名称", "名称", "样品"),
    "mutated_sequence": ("mutated_sequence", "sequence", "突变序列", "全长序列", "序列"),
}

#: 属性名别名 -> 平台指标键
PROPERTY_ALIASES: dict[str, str] = {
    # 热稳定性
    "热稳定性": "thermostability", "热稳定": "thermostability", "耐热性": "thermostability",
    "tm": "thermostability", "t50": "thermostability", "thermostability": "thermostability",
    "half_life": "thermostability", "半衰期": "thermostability",
    # 碱稳定性
    "碱稳定性": "alkali_stability", "耐碱性": "alkali_stability", "碱耐受": "alkali_stability",
    "alkali_stability": "alkali_stability", "alkaline_stability": "alkali_stability",
    # 酸稳定性
    "酸稳定性": "acid_stability", "耐酸性": "acid_stability", "酸耐受": "acid_stability",
    "acid_stability": "acid_stability",
    # 溶解性
    "溶解性": "solubility", "可溶性": "solubility", "溶解度": "solubility",
    "solubility": "solubility",
    # 聚集
    "聚集": "aggregation", "聚集倾向": "aggregation", "聚集风险": "aggregation",
    "aggregation": "aggregation",
    # 表达量
    "表达量": "expression", "表达水平": "expression", "产量": "expression",
    "可溶表达量": "expression", "expression": "expression", "expression_level": "expression",
    "yield": "expression",
    # 酶活
    "酶活": "activity", "比活": "activity", "活性": "activity", "kcat": "activity",
    "kcat_km": "activity", "催化效率": "activity", "activity": "activity",
    # 亲和力
    "亲和力": "affinity", "结合亲和力": "affinity", "kd": "affinity", "affinity": "affinity",
    # 蛋白酶抗性
    "蛋白酶抗性": "protease_resistance", "protease_resistance": "protease_resistance",
    # 免疫原性
    "免疫原性": "immunogenicity", "immunogenicity": "immunogenicity",
    # 修饰位点
    "修饰位点": "ptm_sites", "脱酰胺": "ptm_sites", "氧化": "ptm_sites", "ptm": "ptm_sites",
}

REQUIRED_FIELDS: tuple[str, ...] = ("property_name", "measured_value")

#: 属性值允许的纯数值格式（允许前置比较符与单位后缀，由解析器处理）
_NUMERIC_PATTERN = re.compile(r"^[<>≤≥=~约]*\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


@dataclass
class TemplateColumn:
    """模板列定义。"""

    name: str
    label: str
    required: bool = False
    example: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "required": self.required,
            "example": self.example,
            "description": self.description,
        }


#: 提供给企业的标准录入模板
TEMPLATE_COLUMNS: tuple[TemplateColumn, ...] = (
    TemplateColumn("sequence_name", "序列/样品名称", True, "COL1A1-WT", "同一批实验的样品标识"),
    TemplateColumn("mutation", "突变", False, "A123V", "野生型留空；多点突变用逗号分隔，如 A123V,G456P"),
    TemplateColumn("property_name", "属性", True, "thermostability", "热稳定性/溶解性/表达量 等，支持中英文"),
    TemplateColumn("measured_value", "实测值", True, "68.5", "纯数值；可带比较符（如 >90）"),
    TemplateColumn("unit", "单位", False, "°C", "如 °C、mg/L、U/mg、%"),
    TemplateColumn("condition", "测定条件", False, "pH 7.0, 25 °C", "缓冲液、pH、温度等"),
    TemplateColumn("replicate", "重复", False, "1", "生物学/技术重复编号"),
    TemplateColumn("operator", "操作人", False, "张三", "记录人"),
    TemplateColumn("measured_at", "测定日期", False, "2026-03-15", "ISO 日期"),
    TemplateColumn("note", "备注", False, "0.1M NaOH 处理 30min 后测", "任何补充说明"),
)


def template_columns() -> list[dict[str, Any]]:
    """返回录入模板的列定义。"""
    return [column.to_dict() for column in TEMPLATE_COLUMNS]


def normalize_header(name: Any) -> str:
    """规范化列名：去空格、转小写、去全角括号内容。"""
    text = str(name).strip().lower()
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[\s_\-/]+", "_", text)
    return text.strip("_")


def map_columns(columns: list[Any]) -> tuple[dict[str, str], list[str]]:
    """把文件列名映射到标准字段。

    Returns:
        ``({原始列名: 标准字段}, [未映射的列名])``
    """
    normalized = {str(column): normalize_header(column) for column in columns}
    lookup: dict[str, str] = {}
    for field_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            lookup.setdefault(normalize_header(alias), field_name)

    mapping: dict[str, str] = {}
    unmapped: list[str] = []
    used_fields: set[str] = set()

    for original, norm in normalized.items():
        field_name = lookup.get(norm)
        if field_name is None:
            # 再做一次包含匹配，处理"实测值(°C)"这类带后缀的表头
            for alias_norm, candidate in lookup.items():
                if alias_norm and alias_norm in norm:
                    field_name = candidate
                    break
        if field_name is None or field_name in used_fields:
            unmapped.append(original)
            continue
        mapping[original] = field_name
        used_fields.add(field_name)

    return mapping, unmapped


def normalize_property_name(raw: Any) -> tuple[str, bool]:
    """把属性名归一为平台指标键。

    Returns:
        ``(归一后的名称, 是否为已知指标)``。未知名称原样保留并标记为未知，
        以便企业自行扩展属性而不会被拒绝。
    """
    if raw is None:
        return "", False
    text = str(raw).strip()
    if not text:
        return "", False
    key = normalize_header(text)
    if key in PROPERTY_ALIASES:
        return PROPERTY_ALIASES[key], True
    if text in PROPERTY_ALIASES:
        return PROPERTY_ALIASES[text], True
    return text, False


def parse_numeric(raw: Any) -> tuple[float | None, str | None]:
    """把单元格解析为数值，返回 ``(值, 警告)``。

    支持 ``>90``、``≤0.5``、``约 12`` 这类写法：按字面数值录入，
    但回传警告说明存在不等号，避免被当成精确值。
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw), None

    text = str(raw).strip()
    if not text:
        return None, None
    match = _NUMERIC_PATTERN.match(text)
    if match is None:
        return None, f"无法解析为数值：{text!r}"
    value = float(match.group(1))
    prefix = text[: match.start(1)]
    if prefix.strip(" \t"):
        return value, f"原值 {text!r} 含比较符/修饰词，已按 {value} 录入，请确认"
    if text[match.end(1) :].strip():
        return value, f"原值 {text!r} 含单位后缀，已按 {value} 录入，建议把单位填到 unit 列"
    return value, None


@dataclass
class ParsedRow:
    """解析后的单行数据。"""

    row_number: int
    payload: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unknown_property: bool = False


def read_table(content: bytes, filename: str) -> pd.DataFrame:
    """读取 CSV/Excel 为 DataFrame。"""
    lower = filename.lower()
    if lower.endswith((".xlsx", ".xlsm")):
        return pd.read_excel(io.BytesIO(content), engine="openpyxl", dtype=object)
    if lower.endswith(".xls"):
        raise ValueError("不支持旧版 .xls 格式，请另存为 .xlsx 或 .csv")
    # CSV：优先 utf-8-sig（Excel 导出常带 BOM），失败回退 gbk
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=encoding, dtype=object)
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError as exc:
            raise ValueError(f"CSV 解析失败（分隔符或引号不匹配）：{exc}") from exc
    raise ValueError("CSV 编码无法识别，请另存为 UTF-8 编码")


def parse_dataframe(frame: pd.DataFrame) -> tuple[list[ParsedRow], dict[str, str], list[str]]:
    """逐行解析 DataFrame。"""
    mapping, unmapped = map_columns(list(frame.columns))
    parsed: list[ParsedRow] = []

    field_by_column = {column: field_name for column, field_name in mapping.items()}

    for offset, (_, row) in enumerate(frame.iterrows()):
        row_number = offset + 2  # 1-based 且跳过表头
        item = ParsedRow(row_number=row_number)

        for column, field_name in field_by_column.items():
            value = row[column]
            if field_name == "measured_value":
                number, warning = parse_numeric(value)
                item.payload["measured_value"] = number
                if warning:
                    item.warnings.append(f"第 {row_number} 行：{warning}")
                continue
            if field_name == "replicate":
                number, _ = parse_numeric(value)
                item.payload["replicate"] = int(number) if number is not None else None
                continue
            if value is None or (isinstance(value, float) and pd.isna(value)):
                item.payload[field_name] = None
                continue
            text = str(value).strip()
            item.payload[field_name] = text or None

        # 属性名归一
        raw_property = item.payload.get("property_name")
        normalized, known = normalize_property_name(raw_property)
        item.payload["property_name"] = normalized or None
        item.unknown_property = bool(normalized) and not known

        # 突变标签规范化：去掉空格，统一大写
        mutation = item.payload.get("mutation")
        if mutation:
            item.payload["mutation"] = ",".join(
                part.strip().upper() for part in str(mutation).split(",") if part.strip()
            )
        else:
            item.payload["mutation"] = ""

        # 必填校验
        for field_name in REQUIRED_FIELDS:
            if item.payload.get(field_name) in (None, ""):
                item.errors.append(
                    {
                        "row": row_number,
                        "field": field_name,
                        "message": f"缺少必填字段 {field_name}",
                        "raw": {str(k): str(v) for k, v in row.to_dict().items()},
                    }
                )

        parsed.append(item)

    return parsed, mapping, unmapped


def persist_parsed_rows(
    session,
    rows: list[ParsedRow],
    *,
    project_id: int,
    source_file: str | None = None,
    deduplicate: bool = True,
) -> dict[str, Any]:
    """把通过校验的行写入数据库。

    去重规则：``(sequence_id, mutation, property_name, condition, replicate)`` 完全相同
    的记录视为重复，默认跳过并计数——重复导入同一份 Excel 不会污染训练集。
    """
    from ...db.models import ExperimentRecord

    inserted_ids: list[int] = []
    duplicates = 0

    for item in rows:
        if item.errors:
            continue
        payload = item.payload
        query = session.query(ExperimentRecord).filter(
            ExperimentRecord.property_name == payload["property_name"],
            ExperimentRecord.mutation == (payload.get("mutation") or ""),
            ExperimentRecord.sequence_id.is_(None)
            if payload.get("sequence_id") is None
            else ExperimentRecord.sequence_id == payload.get("sequence_id"),
        )
        condition = payload.get("condition")
        query = (
            query.filter(ExperimentRecord.condition.is_(None))
            if condition is None
            else query.filter(ExperimentRecord.condition == condition)
        )
        replicate = payload.get("replicate")
        query = (
            query.filter(ExperimentRecord.replicate.is_(None))
            if replicate is None
            else query.filter(ExperimentRecord.replicate == replicate)
        )

        if deduplicate and query.first() is not None:
            duplicates += 1
            continue

        measured_at = payload.get("measured_at")
        parsed_date = None
        if measured_at:
            try:
                parsed_date = pd.to_datetime(measured_at).to_pydatetime()
            except Exception:
                item.warnings.append(f"第 {item.row_number} 行：日期 {measured_at!r} 无法解析，已置空")

        record = ExperimentRecord(
            project_id=project_id,
            sequence_id=payload.get("sequence_id") or None,
            design_run_id=payload.get("design_run_id") or None,
            mutation=payload.get("mutation") or "",
            mutated_sequence=payload.get("mutated_sequence") or None,
            property_name=payload["property_name"],
            measured_value=float(payload["measured_value"]),
            unit=payload.get("unit"),
            condition=payload.get("condition"),
            replicate=replicate,
            operator=payload.get("operator"),
            measured_at=parsed_date,
            note=payload.get("note"),
            source_file=source_file,
        )
        session.add(record)
        session.flush()
        inserted_ids.append(record.id)

    logger.info(
        "实验数据入库：新增 %d 条，重复跳过 %d 条（来源 %s）",
        len(inserted_ids),
        duplicates,
        source_file or "未标注",
    )
    return {"inserted_ids": inserted_ids, "duplicate_rows": duplicates}
