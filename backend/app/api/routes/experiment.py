"""实验数据录入、对比分析与模板下载路由。"""

from __future__ import annotations

import io
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session

from ...core.errors import NotFoundError, ValidationError
from ...db.models import ExperimentRecord, Project, ProteinSequence
from ...schemas.common import OkOut, Page
from ...schemas.experiment import (
    ComparisonOut,
    ExperimentRecordIn,
    ExperimentRecordOut,
    IngestResultOut,
    TemplateColumnOut,
)
from ...services.experiment import ingest
from ...services.experiment.compare import ComparisonOptions, compare_records
from ...services.sequence.validator import validate_sequence
from ..deps import get_db

router = APIRouter(prefix="/experiment", tags=["experiment"])


@router.get("/template", response_model=list[TemplateColumnOut], summary="录入模板列定义")
def template_columns() -> list[TemplateColumnOut]:
    """返回标准录入模板的列定义、示例与说明。"""
    return [TemplateColumnOut(**item) for item in ingest.template_columns()]


@router.get("/template.xlsx", summary="下载 Excel 录入模板")
def template_xlsx() -> StreamingResponse:
    """生成可直接发放给实验团队的 Excel 模板（含列说明与示例行）。"""
    import pandas as pd

    columns = [item.to_dict() for item in ingest.TEMPLATE_COLUMNS]
    header = [item["label"] for item in columns]
    example = [item["example"] for item in columns]
    guide = [f"{item['description']}（{'必填' if item['required'] else '选填'}）" for item in columns]

    frame = pd.DataFrame([example, ["", "", "", "", "", "", "", "", "", ""]], columns=header)
    guide_frame = pd.DataFrame([guide], columns=header)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="实验数据", index=False)
        guide_frame.to_excel(writer, sheet_name="列说明", index=False)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="experiment_template.xlsx"'},
    )


@router.get("/template.csv", summary="下载 CSV 录入模板")
def template_csv() -> Response:
    """CSV 版模板（带 BOM，Excel 直接打开不乱码）。"""
    header = [item.label for item in ingest.TEMPLATE_COLUMNS]
    example = [item.example for item in ingest.TEMPLATE_COLUMNS]
    content = "\ufeff" + ",".join(header) + "\n" + ",".join(example) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="experiment_template.csv"'},
    )


@router.post("/ingest", response_model=IngestResultOut, summary="导入实验数据（CSV/Excel）")
async def ingest_file(
    file: UploadFile = File(...),
    project_id: int = Form(default=1),
    sequence_id: int | None = Form(default=None),
    dry_run: bool = Form(default=True, description="true 只校验不写库，用于前端预览"),
    deduplicate: bool = Form(default=True),
    session: Session = Depends(get_db),
) -> IngestResultOut:
    """上传 CSV/Excel 实验数据。

    建议先 ``dry_run=true`` 预览校验结果（列映射、坏行定位），确认无误后再正式导入。
    **绝不静默丢弃数据**：每一行的错误都会带行号与字段返回。
    """
    from ...core.config import get_settings

    settings = get_settings()
    raw = await file.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise ValidationError(f"文件超过 {settings.max_upload_mb} MB 上限")

    try:
        frame = ingest.read_table(raw, file.filename or "upload.csv")
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    if frame.empty:
        raise ValidationError("文件内容为空或没有解析到任何数据行")

    parsed, mapping, unmapped = ingest.parse_dataframe(frame)

    # 补全 project_id / sequence_id
    #
    # 为什么必须在这里解析 sequence_name
    # ----------------------------------
    # CSV 的「序列/样品名称」列此前被解析出来却**从未被使用**，而界面上传默认又只带
    # project_id，于是记录的 sequence_id 全为 NULL。预测-实测对比严格按
    # ``ExperimentRecord.sequence_id == sequence_id`` 过滤，NULL == id 恒不成立，
    # 结果是**界面录入的数据永远无法参与对比**，整个实验回流闭环断裂。
    # 这里让文件具备自描述能力：未显式指定序列时，按行内 sequence_name 在项目内反查。
    #
    # 安全约定：只在精确命中**唯一一条**同名序列时才建立关联。命中 0 条或多条一律留空
    # 并在导入报告中说明——宁可让用户补一次选择，也不能猜错序列，否则后续对比会
    # 把某个突变体的实测值算到另一条序列头上。
    association_notes: list[str] = []
    name_cache: dict[str, list[ProteinSequence]] = {}
    unresolved: dict[str, int] = {}
    elsewhere: dict[str, int] = {}
    resolved_rows = 0

    # 显式指定序列时校验归属
    # ----------------------
    # 此前这里是直接赋值、不做任何检查。后果是记录可以"挂在 A 项目、却引用
    # B 项目的序列"：项目列表与看板按 project_id 统计、预测-实测对比按
    # sequence_id 过滤，两边对"这批数据属于谁"的认知不一致，排查极费劲。
    #
    # 只提示、不阻断：跨项目复用一条参考序列是合法需求，硬拦会挡住正当用法；
    # 但必须让人知道自己在做什么。同一次请求只提示一条，不按行重复。
    if sequence_id is not None:
        explicit_sequence = session.get(ProteinSequence, sequence_id)
        if explicit_sequence is None:
            raise ValidationError(f"序列 id={sequence_id} 不存在")
        if explicit_sequence.project_id != project_id:
            owner = session.get(Project, explicit_sequence.project_id)
            association_notes.append(
                f"所选序列「{explicit_sequence.name}」属于项目「"
                f"{owner.name if owner else explicit_sequence.project_id}」，"
                "与本次导入的目标项目不一致。记录将按当前项目保存，"
                "这会导致项目归属与序列归属不一致（列表与统计按项目过滤，"
                "对比按序列过滤）。如非有意为之，建议切换到该项目后重新导入。"
            )

    for item in parsed:
        item.payload.setdefault("project_id", project_id)
        item.payload["project_id"] = project_id

        if sequence_id is not None:
            item.payload["sequence_id"] = sequence_id
            continue

        raw_name = item.payload.get("sequence_name")
        if not raw_name:
            item.payload["sequence_id"] = None
            continue

        key = str(raw_name).strip()
        if key not in name_cache:
            name_cache[key] = (
                session.query(ProteinSequence)
                .filter(
                    ProteinSequence.project_id == project_id,
                    ProteinSequence.name == key,
                )
                .all()
            )
        matches = name_cache[key]
        if len(matches) == 1:
            item.payload["sequence_id"] = matches[0].id
            resolved_rows += 1
        else:
            item.payload["sequence_id"] = None
            unresolved[key] = len(matches)
            if not matches:
                # 帮用户定位最常见的误操作：序列其实在**别的项目**里。
                # 只说"不存在"会让人反复核对序列名，而真正的问题是项目选错了。
                elsewhere[key] = (
                    session.query(ProteinSequence)
                    .filter(
                        ProteinSequence.project_id != project_id,
                        ProteinSequence.name == key,
                    )
                    .count()
                )

    if resolved_rows:
        association_notes.append(
            f"已按文件内的「序列/样品名称」自动关联 {resolved_rows} 行记录到对应序列。"
        )
    for key, count in sorted(unresolved.items()):
        if count == 0:
            other = elsewhere.get(key, 0)
            if other:
                association_notes.append(
                    f"序列名「{key}」不在当前项目内，但在**其它项目**中存在 {other} 条同名序列；"
                    "相关记录未关联序列，将无法参与预测-实测对比。"
                    "请切换到对应项目后重新导入，或在上传时手动选择序列。"
                )
            else:
                association_notes.append(
                    f"序列名「{key}」在本项目内不存在，相关记录未关联序列，"
                    "将无法参与预测-实测对比；请先保存该序列或在上传时手动选择。"
                )
        else:
            association_notes.append(
                f"序列名「{key}」在本项目内有 {count} 条同名序列，为避免误关联已留空；"
                "请改用上传时手动选择序列。"
            )

    valid_rows = [item for item in parsed if not item.errors]
    invalid_rows = [item for item in parsed if item.errors]

    report_errors: list[dict[str, Any]] = []
    for item in invalid_rows:
        report_errors.extend(item.errors)
        for warning in item.warnings:
            report_errors.append(
                {"row": item.row_number, "field": "measured_value", "message": warning, "raw": {}}
            )

    warnings: list[str] = []
    # 序列关联情况放在最前面：它决定这批数据能否参与预测-实测对比，最关键
    warnings.extend(association_notes)
    for item in parsed:
        warnings.extend(item.warnings)

    unknown = sorted({item.payload.get("property_name") for item in valid_rows if item.unknown_property})
    if unknown:
        warnings.append(
            f"以下属性名不在平台内置指标中，已按原样保留：{'、'.join(unknown)}。"
            "它们可以录入与查询，但无法参与预测-实测对比。"
        )

    property_counts: dict[str, int] = {}
    for item in valid_rows:
        key = str(item.payload.get("property_name"))
        property_counts[key] = property_counts.get(key, 0) + 1

    inserted_ids: list[int] = []
    duplicates = 0
    if not dry_run and valid_rows:
        result = ingest.persist_parsed_rows(
            session,
            valid_rows,
            project_id=project_id,
            source_file=file.filename,
            deduplicate=deduplicate,
        )
        inserted_ids = result["inserted_ids"]
        duplicates = result["duplicate_rows"]
        if duplicates:
            warnings.append(f"有 {duplicates} 条记录与库中已有数据完全一致，已跳过（去重）")

    preview = [
        {
            "row": item.row_number,
            "mutation": item.payload.get("mutation"),
            "property_name": item.payload.get("property_name"),
            "measured_value": item.payload.get("measured_value"),
            "unit": item.payload.get("unit"),
            "condition": item.payload.get("condition"),
            "unknown_property": item.unknown_property,
        }
        for item in valid_rows[:50]
    ]

    from ...schemas.experiment import IngestErrorOut, IngestReportOut

    return IngestResultOut(
        report=IngestReportOut(
            total_rows=len(parsed),
            accepted_rows=len(valid_rows),
            rejected_rows=len(invalid_rows),
            duplicate_rows=duplicates,
            detected_columns={str(original): field for original, field in mapping.items()},
            unmapped_columns=unmapped,
            property_counts=property_counts,
            errors=[IngestErrorOut(**item) for item in report_errors[:200]],
            warnings=warnings[:50],
            dry_run=dry_run,
        ),
        inserted_ids=inserted_ids,
        preview=preview,
    )


@router.get("/records", response_model=Page[ExperimentRecordOut], summary="查询实验记录")
def list_records(
    project_id: int | None = None,
    sequence_id: int | None = None,
    property_name: str | None = None,
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
) -> Page[ExperimentRecordOut]:
    query = session.query(ExperimentRecord)
    if project_id is not None:
        query = query.filter(ExperimentRecord.project_id == project_id)
    if sequence_id is not None:
        query = query.filter(ExperimentRecord.sequence_id == sequence_id)
    if property_name:
        query = query.filter(ExperimentRecord.property_name == property_name)

    total = query.count()
    items = query.order_by(ExperimentRecord.id.desc()).offset(offset).limit(limit).all()
    return Page[ExperimentRecordOut](
        items=[ExperimentRecordOut.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/records", response_model=ExperimentRecordOut, summary="单条录入实验记录")
def create_record(
    payload: ExperimentRecordIn, session: Session = Depends(get_db)
) -> ExperimentRecordOut:
    record = ExperimentRecord(**payload.model_dump())
    session.add(record)
    session.flush()
    return ExperimentRecordOut.model_validate(record)


@router.delete("/records/{record_id}", response_model=OkOut, summary="删除实验记录")
def delete_record(record_id: int, session: Session = Depends(get_db)) -> OkOut:
    record = session.get(ExperimentRecord, record_id)
    if record is None:
        raise NotFoundError(f"实验记录 id={record_id} 不存在")
    session.delete(record)
    return OkOut(message=f"已删除记录 {record_id}")


@router.post("/compare", response_model=ComparisonOut, summary="预测-实测对比分析")
def compare(
    sequence_id: int = Query(description="野生型序列 id"),
    property_name: str | None = Query(default=None, description="只对比指定属性"),
    use_esm: bool = Query(default=False, description="是否用 ESM-2 计算突变体性质（更准但更慢）"),
    protein_type: str = Query(default="generic"),
    host_system: str = Query(default="ecoli"),
    session: Session = Depends(get_db),
) -> ComparisonOut:
    """把实验实测值与平台预测值配对，计算秩相关与（校准后的）误差指标。

    **方法学提示**：平台指标是 0-100 评分，实测值有各自量纲，两者不同尺度，
    因此以 **Spearman 秩相关**为主指标（零样本预测器的标准评估方式）；
    MAE/RMSE/R² 基于一阶线性校准后的预测值，单位与实测一致。
    """
    sequence = session.get(ProteinSequence, sequence_id)
    if sequence is None:
        raise NotFoundError(f"序列 id={sequence_id} 不存在")

    query = session.query(ExperimentRecord).filter(ExperimentRecord.sequence_id == sequence_id)
    if property_name:
        query = query.filter(ExperimentRecord.property_name == property_name)
    records = query.all()

    payload = [
        {
            "mutation": record.mutation,
            "property_name": record.property_name,
            "measured_value": record.measured_value,
            "unit": record.unit,
            "condition": record.condition,
            "replicate": record.replicate,
        }
        for record in records
    ]
    if not payload:
        raise ValidationError(
            f"序列 {sequence.name} 没有可对比的实验记录",
            detail={"hint": "请先在实验与迭代页面导入实测数据"},
        )

    result = compare_records(
        sequence.sequence,
        payload,
        ComparisonOptions(
            use_esm=use_esm, host_system=host_system, protein_type=protein_type
        ),
    )
    result["sequence_id"] = sequence_id
    result["sequence_name"] = sequence.name
    return ComparisonOut(**{key: value for key, value in result.items() if key in ComparisonOut.model_fields})


@router.post("/validate-sequence", summary="校验突变标签是否与序列匹配")
def check_mutation(
    sequence_id: int = Query(...),
    mutation: str = Query(..., description="突变标签，如 A123V，多点用逗号分隔"),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    """录入前校验：突变标签是否与野生型序列一致。

    这一步能挡住最常见的录入错误——把不同构建体的编号混用。
    """
    from ...services.experiment.compare import apply_mutations, parse_mutation_label

    sequence = session.get(ProteinSequence, sequence_id)
    if sequence is None:
        raise NotFoundError(f"序列 id={sequence_id} 不存在")

    details: list[dict[str, Any]] = []
    errors: list[str] = []
    for part in mutation.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            details.append(parse_mutation_label(part, sequence.sequence))
        except ValueError as exc:
            errors.append(str(exc))

    mutated = None
    if not errors:
        try:
            mutated = apply_mutations(sequence.sequence, mutation)
        except ValueError as exc:
            errors.append(str(exc))

    return {
        "ok": not errors,
        "sequence_id": sequence_id,
        "sequence_length": sequence.length,
        "details": details,
        "errors": errors,
        "mutated_sequence_length": len(mutated) if mutated else None,
        "check": validate_sequence(mutated).ok if mutated else False,
    }
