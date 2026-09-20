"""突变设计路由。"""

from __future__ import annotations

import csv
import io
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from ...core.errors import NotFoundError
from ...db.models import DesignRun, MutationCandidate, ProteinSequence
from ...jobs.queue import submit_job
from ...jobs.runner import HEAVY_KINDS, handler_for
from ...schemas.common import JobAccepted, Page
from ...schemas.mutation import DesignRequestIn, ManualPlanRequestIn
from ...services.design.engine import design_catalog
from ...services.design.manual import build_manual_plan, ensure_plan_is_runnable
from ..deps import get_db, resolve_project_id

router = APIRouter(prefix="/design", tags=["design"])


@router.get("/catalog", summary="设计能力目录")
def catalog() -> dict[str, Any]:
    """返回可用的设计模式、蛋白类型、评分维度权重与扫描参数。"""
    return design_catalog()


@router.post("/run", response_model=JobAccepted, summary="提交突变设计作业")
def run(payload: DesignRequestIn, session: Session = Depends(get_db)) -> JobAccepted:
    """提交异步突变设计作业。

    * ``single``：全位点 × 19 氨基酸的单点扫描（一次批量掩码前向）；
    * ``combination``：在单点基础上做束搜索 + 上位效应精评；
    * ``local``：限定区段内只做保守替换；
    * ``manual``：位点、替换氨基酸与锁定位点全部由人工指定，平台只做评分
      （对应"验证 MVP 阶段开发"的人工工作流）。
    """
    project_id = resolve_project_id(session, payload.project_id)
    job_id = submit_job(
        "design",
        handler_for("design"),
        {
            "sequence": payload.sequence,
            "protein_type": payload.protein_type,
            "mode": payload.mode,
            "host_system": payload.host_system,
            "region": payload.region,
            "max_positions": payload.max_positions,
            "top_n": payload.top_n,
            "max_sites": payload.max_sites,
            "beam_width": payload.beam_width,
            "use_structure": payload.use_structure,
            "provider": payload.provider,
            "name": payload.name,
            "project_id": project_id,
            # 人工模式参数（其它模式忽略）
            "target_positions": payload.target_positions,
            "substitutions": payload.substitutions,
            "locked_positions": payload.locked_positions,
            "mutation_notes": payload.mutation_notes,
        },
        project_id=project_id,
        title=f"突变设计（{payload.protein_type} / {payload.mode}）",
        heavy="design" in HEAVY_KINDS,
    )
    return JobAccepted(job_id=job_id)


@router.post("/manual/plan", summary="预览人工突变清单（不触发评估）")
def manual_plan_preview(payload: ManualPlanRequestIn) -> dict[str, Any]:
    """按人工给定的位点与替换氨基酸，展开成突变清单。

    该接口**只做规划与校验**：不调用 ESM-2、不预测结构、不写数据库，
    因此同步返回、毫秒级完成。用途是让用户在真正提交（要跑结构与打分）之前，
    先把"平台将评估哪些突变"看一遍。

    与提交路径的关键差异：**这里允许带冲突返回**。``blocked`` 字段会逐条列出
    被拒绝的位点与原因（锁定位点、规则包保护、末端等），前端把问题摆给用户改；
    而提交路径一旦发现冲突就直接拒绝执行——因为位点是人指定的，
    少评一条就等于"人以为在评的"与"平台实际评的"不一致。
    """
    plan = build_manual_plan(
        payload.sequence,
        target_positions=payload.target_positions,
        substitutions=payload.substitutions,
        locked_positions=payload.locked_positions,
        protein_type=payload.protein_type,
        name=payload.name,
        notes=payload.mutation_notes,
    )
    return plan.to_dict()


@router.post("/manual/export", summary="导出人工突变清单（FASTA / CSV）")
def manual_plan_export(
    payload: ManualPlanRequestIn,
    format: str = Query(default="csv", pattern="^(csv|fasta)$"),
) -> Response:
    """导出突变清单，供湿实验合成基因与登记使用。

    CSV 的前三列与需求要求的 ``sequence_id, fasta, mutation_note`` 一致，
    其余列为便于人工核对而附加。默认把**野生型放在最前面作为基准对照**
    （需求"风险控制要点"第 2 条：所有指标都是相对野生对比）。

    导出即视为清单定稿，因此这里会执行与提交路径相同的严格校验：
    只要有一个目标位点被拒绝就报错，而不是导出一份"少了几个位点"的清单。
    """
    plan = build_manual_plan(
        payload.sequence,
        target_positions=payload.target_positions,
        substitutions=payload.substitutions,
        locked_positions=payload.locked_positions,
        protein_type=payload.protein_type,
        name=payload.name,
        notes=payload.mutation_notes,
    )
    ensure_plan_is_runnable(plan)

    if format == "fasta":
        body = plan.to_fasta(payload.include_wild_type)
        media_type = "chemical/x-fasta; charset=utf-8"
        filename = f"{plan.name}_mutations.fasta"
    else:
        body = plan.to_csv(payload.include_wild_type)
        media_type = "text/csv; charset=utf-8"
        filename = f"{plan.name}_mutations.csv"

    return Response(
        content=body.encode("utf-8"),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/runs", response_model=Page[dict], summary="列出设计批次")
def list_runs(
    sequence_id: int | None = None,
    category: str | None = None,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
) -> Page[dict]:
    query = session.query(DesignRun)
    if sequence_id is not None:
        query = query.filter(DesignRun.sequence_id == sequence_id)
    if category:
        query = query.filter(DesignRun.category == category)
    total = query.count()
    items = query.order_by(DesignRun.id.desc()).offset(offset).limit(limit).all()

    payload: list[dict] = []
    for record in items:
        sequence = session.get(ProteinSequence, record.sequence_id)
        payload.append(
            {
                "id": record.id,
                "sequence_id": record.sequence_id,
                "sequence_name": sequence.name if sequence else None,
                "sequence_length": sequence.length if sequence else None,
                "category": record.category,
                "mode": record.mode,
                "params": record.params,
                "summary": record.summary,
                "created_at": record.created_at,
            }
        )
    return Page[dict](items=payload, total=total, limit=limit, offset=offset)


@router.get("/runs/{run_id}", summary="设计批次详情")
def get_run(run_id: int, session: Session = Depends(get_db)) -> dict[str, Any]:
    record = session.get(DesignRun, run_id)
    if record is None:
        raise NotFoundError(f"设计批次 id={run_id} 不存在")
    sequence = session.get(ProteinSequence, record.sequence_id)
    return {
        "id": record.id,
        "sequence_id": record.sequence_id,
        "sequence_name": sequence.name if sequence else None,
        "sequence": sequence.sequence if sequence else None,
        "category": record.category,
        "mode": record.mode,
        "params": record.params,
        "summary": record.summary,
        "created_at": record.created_at,
    }


@router.get("/runs/{run_id}/candidates", summary="候选方案列表")
def list_candidates(
    run_id: int,
    limit: int = Query(default=100, ge=1, le=5000),
    min_score: float | None = None,
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    """从数据库读取该批次的候选方案（已按排名排序）。"""
    run = session.get(DesignRun, run_id)
    if run is None:
        raise NotFoundError(f"设计批次 id={run_id} 不存在")

    query = session.query(MutationCandidate).filter(MutationCandidate.design_run_id == run_id)
    if min_score is not None:
        query = query.filter(MutationCandidate.total_score >= min_score)
    total = query.count()
    items = query.order_by(MutationCandidate.rank).limit(limit).all()

    return {
        "design_run_id": run_id,
        "total": total,
        "items": [
            {
                "rank": item.rank,
                "mutations": item.mutations,
                "positions": item.positions,
                "total_score": item.total_score,
                "dimensions": item.dimensions,
                "flags": item.flags,
                "rationale": item.rationale,
            }
            for item in items
        ],
    }


@router.get("/runs/{run_id}/export", summary="导出候选方案")
def export_run(
    run_id: int,
    format: str = Query(default="csv", pattern="^(csv|markdown)$"),
    top_n: int = Query(default=20, ge=1, le=2000),
    session: Session = Depends(get_db),
) -> Response:
    """把候选方案导出为 CSV 或 Markdown，可直接交给实验团队。

    导出内容包含**逐维度分数与依据**，而不是只有一个总分——这是需求文档
    "便于实验人员优先筛选"的直接体现。
    """
    run = session.get(DesignRun, run_id)
    if run is None:
        raise NotFoundError(f"设计批次 id={run_id} 不存在")
    sequence = session.get(ProteinSequence, run.sequence_id)

    candidates = (
        session.query(MutationCandidate)
        .filter(MutationCandidate.design_run_id == run_id)
        .order_by(MutationCandidate.rank)
        .limit(top_n)
        .all()
    )
    if not candidates:
        raise NotFoundError(f"批次 {run_id} 没有候选方案记录")

    dimension_keys = ["stability", "activity", "foldability", "expression", "risk"]
    dimension_labels = {
        "stability": "稳定性贡献",
        "activity": "活性影响",
        "foldability": "折叠可行性",
        "expression": "宿主表达适配",
        "risk": "新增风险位点",
    }

    if format == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "排名",
                "突变",
                "位点数",
                "总分",
                *[dimension_labels[key] for key in dimension_keys],
                "风险告警",
                "评分说明",
            ]
        )
        for item in candidates:
            scores = {
                entry.get("key"): entry.get("raw") for entry in (item.dimensions or [])
            }
            writer.writerow(
                [
                    item.rank,
                    ",".join(item.mutations or []),
                    len(item.mutations or []),
                    round(item.total_score, 2),
                    *[round(scores.get(key, 0.0), 2) for key in dimension_keys],
                    "；".join(item.flags or []),
                    (item.rationale or "").replace("\n", " "),
                ]
            )
        content = "\ufeff" + buffer.getvalue()  # BOM，Excel 打开不乱码
        filename = f"design_run_{run_id}_candidates.csv"
        return Response(
            content=content.encode("utf-8"),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    lines = [
        f"# 突变设计候选方案（批次 {run_id}）",
        "",
        f"- 目标蛋白：{sequence.name if sequence else '未知'}（{sequence.length if sequence else '?'} aa）",
        f"- 蛋白类型：{run.category}",
        f"- 设计模式：{run.mode}",
        f"- 候选数量：{len(candidates)}",
        "",
        "| 排名 | 突变 | 总分 | 稳定性 | 活性 | 折叠 | 表达 | 安全性 | 告警 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in candidates:
        scores = {entry.get("key"): entry.get("raw") for entry in (item.dimensions or [])}
        lines.append(
            "| {rank} | {muts} | {total:.1f} | {stability:.0f} | {activity:.0f} | "
            "{foldability:.0f} | {expression:.0f} | {risk:.0f} | {flags} |".format(
                rank=item.rank,
                muts=", ".join(item.mutations or []),
                total=item.total_score,
                stability=scores.get("stability", 0.0),
                activity=scores.get("activity", 0.0),
                foldability=scores.get("foldability", 0.0),
                expression=scores.get("expression", 0.0),
                risk=scores.get("risk", 0.0),
                flags="；".join(item.flags or []) or "—",
            )
        )
    lines.extend(["", "## 逐条说明", ""])
    for item in candidates:
        lines.append(f"### {item.rank}. {', '.join(item.mutations or [])}")
        lines.append("")
        lines.append(item.rationale or "无")
        for entry in item.dimensions or []:
            lines.append(
                f"- **{entry.get('label')}**：{entry.get('raw'):.1f} 分"
                f"（权重 {entry.get('weight', 0):.3f}，贡献 {entry.get('contribution', 0):+.2f}）"
                f" — {entry.get('rationale', '')}"
            )
        lines.append("")

    filename = f"design_run_{run_id}_candidates.md"
    return Response(
        content="\n".join(lines).encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/runs/{run_id}", summary="删除设计批次")
def delete_run(run_id: int, session: Session = Depends(get_db)) -> dict[str, Any]:
    record = session.get(DesignRun, run_id)
    if record is None:
        raise NotFoundError(f"设计批次 id={run_id} 不存在")
    session.delete(record)
    return {"ok": True, "message": f"已删除批次 {run_id}"}
