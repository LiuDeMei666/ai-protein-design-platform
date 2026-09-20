"""作业执行器：把四类长耗时任务包装成作业处理器。

四类作业
--------
==========  ==================================================  ============
kind        内容                                                重计算
==========  ==================================================  ============
structure   结构预测 + 结构统计 + 落库缓存                       是
property    理化性质预测（可选先预测结构 + ESM-2 自然度）        是
design      突变设计（零样本扫描 + 五维评分 + 组合搜索）          是
train       属性头训练 / 增量训练                                否
==========  ==================================================  ============

约定
----
* 作业结果里**不放大体积数据**（PDB 文本单独存取，通过 ``structure_id`` 引用）。
* 每个作业都会把领域结果落库（StructureRecord / PropertyRun / DesignRun），
  作业表只保存摘要与引用，避免重复存储。
* 一切外部依赖失败都抛 :class:`PlatformError` 子类，由队列统一转成 failed 状态。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.errors import NotFoundError, ValidationError
from ..core.logging import describe_sequence, get_logger
from ..db.base import session_scope
from ..db.models import (
    DesignRun,
    MutationCandidate,
    ProteinSequence,
    PropertyRun,
    StructureRecord,
)
from .queue import JobContext

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# 公共工具
# --------------------------------------------------------------------------- #
def ensure_sequence(
    session: Session,
    *,
    project_id: int,
    sequence: str,
    protein_type: str = "generic",
    name: str | None = None,
) -> ProteinSequence:
    """按 (project_id, sha256) 取或建序列记录。"""
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    existing = (
        session.query(ProteinSequence)
        .filter(
            ProteinSequence.project_id == project_id,
            ProteinSequence.sha256 == digest,
        )
        .one_or_none()
    )
    if existing is not None:
        # 蛋白类型可能被用户重新指定，以最新一次为准
        if protein_type and existing.protein_type != protein_type:
            existing.protein_type = protein_type
        return existing

    record = ProteinSequence(
        project_id=project_id,
        name=name or f"未命名序列 {len(sequence)}aa",
        protein_type=protein_type or "generic",
        sequence=sequence,
        length=len(sequence),
        sha256=digest,
    )
    session.add(record)
    session.flush()
    return record


def _persist_structure(session: Session, sequence: str, result: Any) -> StructureRecord:
    """把结构结果落库 + PDB 落盘。"""
    settings = get_settings()
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    pdb_dir = Path(settings.cache_dir) / "pdb_models"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = pdb_dir / f"{digest[:16]}_{result.source}.pdb"
    pdb_path.write_text(result.pdb_text, encoding="utf-8")

    existing = (
        session.query(StructureRecord)
        .filter(
            StructureRecord.sequence_sha256 == digest,
            StructureRecord.provider == result.source,
        )
        .one_or_none()
    )
    payload = {
        "sequence_sha256": digest,
        "sequence_length": len(sequence),
        "provider": result.source,
        "model_version": result.model_version,
        "pdb_path": str(pdb_path),
        "plddt": [round(float(value), 2) for value in result.plddt],
        "mean_plddt": float(result.mean_plddt),
        "truncated": bool(result.truncated),
        "segments": [list(segment) for segment in result.segments],
        "secondary_structure": (result.stats or {}).get("secondary_structure", ""),
        "stats": _strip_large_stats(result.stats or {}),
        "degradation_reason": result.degradation_reason,
    }
    if existing is None:
        existing = StructureRecord(**payload)
        session.add(existing)
    else:
        for key, value in payload.items():
            setattr(existing, key, value)
    session.flush()
    return existing


def _strip_large_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """结构统计中剔除超大字段（接触图另存），避免数据库膨胀。"""
    payload = dict(stats)
    contacts = payload.get("contacts")
    if contacts:
        # 接触图限制在 8000 对以内
        payload["contacts"] = contacts[:8000]
        payload["contacts_truncated"] = len(contacts) > 8000
    return payload


def _structure_payload(result: Any, structure_id: int | None) -> dict[str, Any]:
    """构造作业结果中的结构摘要。"""
    return {
        "structure_id": structure_id,
        "source": result.source,
        "model_version": result.model_version,
        "mean_plddt": round(float(result.mean_plddt), 2),
        "length": result.length,
        "truncated": result.truncated,
        "segments": [list(segment) for segment in result.segments],
        "from_cache": result.from_cache,
        "degradation_reason": result.degradation_reason,
        "warnings": list(result.warnings),
        "plddt": [round(float(value), 2) for value in result.plddt],
        "secondary_structure": (result.stats or {}).get("secondary_structure", ""),
        "stats": _strip_large_stats(result.stats or {}),
    }


# --------------------------------------------------------------------------- #
# 结构预测作业
# --------------------------------------------------------------------------- #
def structure_job(params: dict[str, Any], context: JobContext) -> dict[str, Any]:
    """结构预测作业。"""
    from ..services.structure.registry import predict_structure
    from ..services.sequence.validator import validate_sequence

    sequence = params.get("sequence") or ""
    context.progress(0.05, "校验序列")
    check = validate_sequence(sequence)
    if not check.ok:
        raise ValidationError("; ".join(check.errors), detail={"warnings": check.warnings})
    sequence = check.sequence

    context.progress(0.15, "预测三维结构")
    context.log(f"开始折叠 {describe_sequence(sequence)}")

    result = predict_structure(
        sequence,
        provider=params.get("provider"),
        allow_split=bool(params.get("allow_split", True)),
        use_cache=bool(params.get("use_cache", True)),
    )

    context.progress(0.8, "解析结构与统计")
    warnings = list(result.warnings)
    if result.degradation_reason:
        warnings.append(result.degradation_reason)

    structure_id: int | None = None
    project_id = params.get("project_id")
    if project_id:
        with session_scope() as session:
            record = _persist_structure(session, sequence, result)
            structure_id = record.id
            ensure_sequence(
                session,
                project_id=int(project_id),
                sequence=sequence,
                protein_type=str(params.get("protein_type") or "generic"),
                name=params.get("name"),
            )

    context.progress(1.0, "完成")
    return {
        "sequence_length": len(sequence),
        "sequence_fingerprint": check.sha256[:12],
        "warnings": warnings,
        **_structure_payload(result, structure_id),
    }


# --------------------------------------------------------------------------- #
# 性质预测作业
# --------------------------------------------------------------------------- #
def property_job(params: dict[str, Any], context: JobContext) -> dict[str, Any]:
    """理化性质预测作业。"""
    from ..services.property.pipeline import predict_properties
    from ..services.sequence.validator import validate_sequence
    from ..services.structure.registry import predict_structure

    sequence = params.get("sequence") or ""
    context.progress(0.05, "校验序列")
    check = validate_sequence(sequence)
    if not check.ok:
        raise ValidationError("; ".join(check.errors))
    sequence = check.sequence

    project_id = params.get("project_id")
    structure_stats: dict[str, Any] | None = None
    structure_payload: dict[str, Any] | None = None
    if params.get("use_structure", True):
        context.progress(0.12, "预测三维结构（用于结构相关指标）")
        try:
            structure = predict_structure(sequence, provider=params.get("provider"))
            is_stub = bool(structure.stats.get("stub"))
            structure_stats = None if is_stub else structure.stats

            structure_id: int | None = None
            if project_id:
                with session_scope() as session:
                    structure_id = _persist_structure(session, sequence, structure).id

            structure_payload = {
                "structure_id": structure_id,
                "source": structure.source,
                "mean_plddt": round(float(structure.mean_plddt), 2),
                "segments": [list(segment) for segment in structure.segments],
                "truncated": structure.truncated,
                "degradation_reason": structure.degradation_reason,
                "warnings": list(structure.warnings),
            }
            if is_stub:
                structure_payload["note"] = "占位结构不计入性质评分"
        except Exception as exc:
            # 结构失败不应让整个性质预测失败：无结构也能给出 9 项指标
            context.log(f"结构预测失败，将以无结构模式继续：{exc}")
            structure_stats = None

    context.progress(0.5, "计算理化性质指标")
    report = predict_properties(
        sequence,
        protein_type=str(params.get("protein_type") or "generic"),
        host_system=str(params.get("host_system") or "ecoli"),
        structure_stats=structure_stats,
        dna_sequence=params.get("dna_sequence"),
        use_esm=bool(params.get("use_esm", True)),
        validator_result=check,
    )

    context.progress(0.9, "保存结果")
    property_run_id: int | None = None
    project_id = params.get("project_id")
    if project_id:
        with session_scope() as session:
            record = ensure_sequence(
                session,
                project_id=int(project_id),
                sequence=sequence,
                protein_type=str(params.get("protein_type") or "generic"),
                name=params.get("name"),
            )
            payload = report.to_dict(include_tracks=False)
            run = PropertyRun(
                sequence_id=record.id,
                job_id=context.job_id,
                host_system=report.host_system,
                use_structure=report.structure_used,
                summary=report.summary,
                results=payload,
            )
            session.add(run)
            session.flush()
            property_run_id = run.id
            sequence_id = record.id
    else:
        sequence_id = None

    context.progress(1.0, "完成")
    payload = report.to_dict(include_tracks=True)
    payload["property_run_id"] = property_run_id
    payload["sequence_id"] = sequence_id
    payload["structure"] = structure_payload
    return payload


# --------------------------------------------------------------------------- #
# 突变设计作业
# --------------------------------------------------------------------------- #
def design_job(params: dict[str, Any], context: JobContext) -> dict[str, Any]:
    """突变设计作业。"""
    from ..services.design.engine import DesignRequest, run_design
    from ..services.sequence.validator import validate_sequence
    from ..services.structure.registry import predict_structure

    sequence = params.get("sequence") or ""
    context.progress(0.03, "校验序列")
    check = validate_sequence(sequence, strict=True)
    if not check.ok:
        raise ValidationError("; ".join(check.errors), detail={"warnings": check.warnings})
    sequence = check.sequence

    project_id = params.get("project_id")
    structure_stats: dict[str, Any] | None = None
    structure_payload: dict[str, Any] | None = None
    if params.get("use_structure", True):
        context.progress(0.08, "预测三维结构（提供位点级结构先验）")
        try:
            structure = predict_structure(sequence, provider=params.get("provider"))
            is_stub = bool(structure.stats.get("stub"))
            structure_stats = None if is_stub else structure.stats

            # 结构落库：前端需要用它在 3D 视图上高亮突变热点（与候选表联动）。
            # 不落库就只能展示统计数字，"热点联动"这一关键交互会失效。
            structure_id: int | None = None
            if project_id:
                with session_scope() as session:
                    structure_id = _persist_structure(session, sequence, structure).id

            structure_payload = {
                "structure_id": structure_id,
                "source": structure.source,
                "mean_plddt": round(float(structure.mean_plddt), 2),
                "segments": [list(segment) for segment in structure.segments],
                "truncated": structure.truncated,
                "degradation_reason": structure.degradation_reason,
                "note": "占位结构未用于打分" if is_stub else "",
            }
        except Exception as exc:
            context.log(f"结构预测失败，将以无结构模式继续：{exc}")

    region = params.get("region")
    region_tuple: tuple[int, int] | None = None
    if isinstance(region, (list, tuple)) and len(region) == 2:
        # 前端给的是 1-based 闭区间，内部用 0-based 半开
        region_tuple = (max(0, int(region[0]) - 1), int(region[1]))

    context.progress(0.15, "计算 ESM-2 零样本突变打分")
    design_request = DesignRequest(
        sequence=sequence,
        protein_type=str(params.get("protein_type") or "generic"),
        mode=str(params.get("mode") or "single"),
        host_system=str(params.get("host_system") or "ecoli"),
        region=region_tuple,
        max_positions=params.get("max_positions"),
        top_n=int(params.get("top_n") or 50),
        max_sites=int(params.get("max_sites") or 3),
        beam_width=int(params.get("beam_width") or 8),
        structure_stats=structure_stats,
        # 人工模式参数；自动模式下这些字段被引擎忽略
        target_positions=params.get("target_positions"),
        substitutions=params.get("substitutions"),
        locked_positions=params.get("locked_positions"),
        mutation_notes=params.get("mutation_notes"),
        name=params.get("name"),
    )
    result = run_design(design_request)

    context.progress(0.88, "保存候选方案")
    design_run_id: int | None = None
    sequence_id: int | None = None
    project_id = params.get("project_id")
    if project_id:
        with session_scope() as session:
            record = ensure_sequence(
                session,
                project_id=int(project_id),
                sequence=sequence,
                protein_type=design_request.protein_type,
                name=params.get("name"),
            )
            sequence_id = record.id
            run = DesignRun(
                sequence_id=record.id,
                job_id=context.job_id,
                category=result.protein_type,
                mode=result.mode,
                params={
                    "region": list(region) if region else None,
                    "max_sites": design_request.max_sites,
                    "beam_width": design_request.beam_width,
                    "top_n": design_request.top_n,
                    "use_structure": structure_stats is not None,
                    # 人工模式必须把"人给了什么"原样落库：后续复现或核对
                    # "为什么只评了这几条"时，只能靠这份记录，结果里推不出来。
                    "manual": (
                        {
                            "target_positions": design_request.target_positions,
                            "locked_positions": design_request.locked_positions,
                            "substitutions": design_request.substitutions,
                            "mutation_count": len(result.manual.get("mutations") or []),
                        }
                        if design_request.mode == "manual"
                        else None
                    ),
                },
                summary=result.summary,
            )
            session.add(run)
            session.flush()
            design_run_id = run.id

            # 只持久化 Top 200 候选，避免数据库膨胀（完整结果在作业结果中）
            for rank, candidate in enumerate(result.candidates[:200], start=1):
                session.add(
                    MutationCandidate(
                        design_run_id=run.id,
                        rank=rank,
                        mutations=candidate.mutations,
                        positions=[position + 1 for position in candidate.positions],
                        total_score=candidate.total_score,
                        dimensions=[item.to_dict() for item in candidate.dimensions],
                        flags=candidate.flags,
                        rationale=candidate.rationale,
                    )
                )

    context.progress(1.0, "完成")
    payload = result.to_dict()
    payload["design_run_id"] = design_run_id
    payload["sequence_id"] = sequence_id
    payload["structure"] = structure_payload
    return payload


# --------------------------------------------------------------------------- #
# 模型训练作业
# --------------------------------------------------------------------------- #
def train_job(params: dict[str, Any], context: JobContext) -> dict[str, Any]:
    """属性头训练 / 增量训练作业。"""
    from ..ml.registry import to_payload
    from ..ml.train import incremental_train, train_property_head

    property_name = str(params.get("property_name") or "").strip()
    if not property_name:
        raise ValidationError("必须指定 property_name")

    incremental = bool(params.get("incremental", False))
    context.progress(0.1, "构建训练集")
    context.log(f"{'增量' if incremental else '全量'}训练属性：{property_name}")

    with session_scope() as session:
        if incremental:
            outcome = incremental_train(session, property_name)
        else:
            outcome = train_property_head(
                session,
                property_name,
                algo=str(params.get("algo") or "ridge"),
                min_samples=int(params.get("min_samples") or 8),
                test_ratio=float(params.get("test_ratio", 0.25)),
                cv_folds=int(params.get("cv_folds") or 5),
                notes=params.get("notes"),
            )
        context.progress(0.9, "注册模型版本")
        version_payload = to_payload(outcome["version"])

    context.progress(1.0, "完成")
    payload = {
        "property_name": property_name,
        "incremental": incremental,
        "version": version_payload,
        "metrics": outcome["metrics"],
    }
    for key in (
        "feature_dim",
        "train_samples",
        "test_samples",
        "warnings",
        "notes",
        "new_samples",
        "total_samples",
        "base_version",
        "algorithm",
        "incremental_equivalent_to_full_refit",
        "alpha",
    ):
        if key in outcome:
            payload[key] = outcome[key]
    return payload


#: 作业类型 -> 处理器
HANDLERS = {
    "structure": structure_job,
    "property": property_job,
    "design": design_job,
    "train": train_job,
}

#: 哪些作业属于"重计算"（需要 GPU 串行化）
HEAVY_KINDS = {"structure", "property", "design"}


def handler_for(kind: str):
    """取作业处理器。"""
    if kind not in HANDLERS:
        raise ValidationError(f"未知的作业类型: {kind}", detail={"available": sorted(HANDLERS)})
    return HANDLERS[kind]
