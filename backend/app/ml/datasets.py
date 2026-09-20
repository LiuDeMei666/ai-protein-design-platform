"""从数据库构建训练集。

数据来源是 :class:`ExperimentRecord`：企业录入的"突变体 -> 实测性质"记录。
本模块负责把这些记录变成 ``(X = ESM-2 嵌入, y = 实测值)`` 的监督学习配对。

关键处理
--------
1. **突变序列还原**：记录可能只给了突变标签（``A123V``）而没有全长序列，
   此时需要结合其所属的野生型序列把突变应用上去。
   无法还原的记录会被**逐条列出原因**，不会被静默丢弃。
2. **去重**：同一个序列可能有多条重复测定（不同重复/条件）。默认取均值聚合，
   并记录原始条数，让用户知道有效样本量。
3. **嵌入缓存**：ESM-2 嵌入按序列哈希落盘缓存，重复训练零成本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from ..core.logging import describe_sequence, get_logger
from ..db.models import ExperimentRecord, ProteinSequence
from ..services.experiment.compare import apply_mutations
from ..services.embedding.esm2 import embed_sequence

logger = get_logger(__name__)


@dataclass
class DatasetBuildResult:
    """训练集构建结果。"""

    property_name: str
    sequences: list[str] = field(default_factory=list)
    embeddings: np.ndarray | None = None
    targets: np.ndarray | None = None
    labels: list[str] = field(default_factory=list)
    record_ids: list[list[int]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return 0 if self.targets is None else int(len(self.targets))

    @property
    def n_features(self) -> int:
        return 0 if self.embeddings is None else int(self.embeddings.shape[1])

    @property
    def is_usable(self) -> bool:
        return self.n_samples > 0 and self.n_features > 0

    def describe(self) -> dict[str, Any]:
        return {
            "property_name": self.property_name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "skipped_count": len(self.skipped),
            "skipped": self.skipped[:50],
            "warnings": self.warnings,
            "labels": self.labels,
        }


def _resolve_target_sequence(
    record: ExperimentRecord, sequence_cache: dict[int, str | None]
) -> tuple[str | None, str | None]:
    """还原一条实验记录对应的蛋白序列。

    Returns:
        ``(序列, 失败原因)``。
    """
    if record.mutated_sequence:
        return record.mutated_sequence.strip().upper(), None

    if record.sequence_id is None:
        return None, "既没有突变序列，也没有关联网序列，无法构建特征"

    if record.sequence_id not in sequence_cache:
        return None, f"关联的序列 id={record.sequence_id} 不存在"

    wild_type = sequence_cache[record.sequence_id]
    if not wild_type:
        return None, f"关联的序列 id={record.sequence_id} 不存在"

    mutation = (record.mutation or "").strip()
    if not mutation:
        return wild_type, None
    try:
        return apply_mutations(wild_type, mutation), None
    except ValueError as exc:
        return None, f"突变标签无法应用：{exc}"


def build_dataset(
    session: Session,
    property_name: str,
    *,
    aggregate: str = "mean",
    min_records: int = 2,
) -> DatasetBuildResult:
    """为指定属性构建训练集。

    Args:
        session: 数据库会话。
        property_name: 目标属性（与实验记录的 ``property_name`` 对齐）。
        aggregate: 同一序列多条记录时的聚合方式：``mean`` 或 ``median``。
        min_records: 至少需要多少条可用样本才继续计算嵌入。
    """
    result = DatasetBuildResult(property_name=property_name)

    records = (
        session.query(ExperimentRecord)
        .filter(ExperimentRecord.property_name == property_name)
        .order_by(ExperimentRecord.id)
        .all()
    )
    if not records:
        result.warnings.append(f"没有任何属性为 {property_name} 的实验记录")
        return result

    # 预取涉及的野生型序列
    sequence_ids = {record.sequence_id for record in records if record.sequence_id is not None}
    sequence_cache: dict[int, str | None] = {}
    if sequence_ids:
        for item in session.query(ProteinSequence).filter(ProteinSequence.id.in_(sequence_ids)).all():
            sequence_cache[item.id] = item.sequence

    # 序列 -> [(值, 记录 id)]
    grouped: dict[str, list[tuple[float, int]]] = {}
    for record in records:
        sequence, reason = _resolve_target_sequence(record, sequence_cache)
        if sequence is None:
            result.skipped.append({"record_id": record.id, "reason": reason})
            continue
        grouped.setdefault(sequence, []).append((float(record.measured_value), record.id))

    if len(grouped) < min_records:
        result.warnings.append(
            f"可用的唯一序列仅 {len(grouped)} 条，少于最小样本数 {min_records}，"
            "数据量不足以训练属性头。建议继续积累实验数据。"
        )
        return result

    sequences: list[str] = []
    targets: list[float] = []
    labels: list[str] = []
    record_ids: list[list[int]] = []

    for sequence, values in grouped.items():
        numbers = np.array([item[0] for item in values], dtype=np.float64)
        aggregated = float(np.median(numbers) if aggregate == "median" else numbers.mean())
        sequences.append(sequence)
        targets.append(aggregated)
        labels.append(describe_sequence(sequence))
        record_ids.append([item[1] for item in values])
        if len(values) > 1:
            result.warnings.append(
                f"序列 {describe_sequence(sequence)} 有 {len(values)} 条重复测定，"
                f"已按 {aggregate} 聚合为 {aggregated:.4f}"
            )

    # 计算嵌入
    embeddings: list[np.ndarray] = []
    for sequence in sequences:
        try:
            embeddings.append(embed_sequence(sequence).mean)
        except Exception as exc:
            result.skipped.append(
                {"sequence": describe_sequence(sequence), "reason": f"嵌入计算失败：{exc}"}
            )

    kept = len(embeddings)
    if kept < min_records:
        result.warnings.append(f"成功计算嵌入的样本仅 {kept} 条，不足以训练。")
        return result

    result.sequences = sequences[:kept]
    result.targets = np.asarray(targets[:kept], dtype=np.float64)
    result.labels = labels[:kept]
    result.record_ids = record_ids[:kept]
    result.embeddings = np.vstack(embeddings).astype(np.float32)

    logger.info(
        "训练集构建完成：属性=%s 样本=%d 特征=%d 跳过=%d",
        property_name,
        result.n_samples,
        result.n_features,
        len(result.skipped),
    )
    return result


def available_properties(session: Session, min_records: int = 1) -> list[dict[str, Any]]:
    """列出可训练的属性及其样本量（供前端展示"哪些属性可以训练"）。

    "有效样本数"按 ``(sequence_id, mutation)`` 去重统计，而不是
    ``count(distinct mutated_sequence)``——后者在"只给突变标签、不给全长序列"
    的常规录入方式下恒为 0（字段为 NULL，SQL 的 DISTINCT 会忽略 NULL），
    会让界面误报"没有可用样本"。
    """
    from sqlalchemy import func

    rows = (
        session.query(
            ExperimentRecord.property_name,
            func.count(ExperimentRecord.id),
            func.count(func.distinct(ExperimentRecord.sequence_id)),
        )
        .group_by(ExperimentRecord.property_name)
        .all()
    )

    payload: list[dict[str, Any]] = []
    for property_name, count, distinct_sequences in rows:
        # 再按 (sequence_id, mutation) 组合精确统计一次
        pairs = (
            session.query(ExperimentRecord.sequence_id, ExperimentRecord.mutation)
            .filter(ExperimentRecord.property_name == property_name)
            .distinct()
            .count()
        )
        payload.append(
            {
                "property_name": property_name,
                "record_count": int(count),
                "distinct_sequences": int(distinct_sequences),
                "distinct_samples": int(pairs),
                "trainable": int(pairs) >= min_records,
            }
        )
    payload.sort(key=lambda item: -item["record_count"])
    return payload
