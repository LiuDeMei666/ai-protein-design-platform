"""属性头训练编排。

流程::

    构建数据集（ESM-2 嵌入 + 实测值）
      -> 划分训练/测试集（按序列去重后随机划分，固定随机种子保证可复现）
      -> 交叉验证选择正则强度（Ridge）
      -> 训练
      -> 在留出集上评估
      -> 保存工件 + 注册模型版本

样本量警示
----------
企业实验数据初期通常只有个位数到几十条。此时：
* 划分留出集会让训练集更小，评估指标波动很大；
* 因此当样本数 < 15 时**不划分留出集**，只用交叉验证指标，并在 notes 里明确说明。
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.errors import ValidationError
from ..core.logging import get_logger
from ..db.models import ExperimentRecord, ModelVersion
from .datasets import build_dataset
from .head import PropertyHead, regression_metrics
from .registry import ALGO_ARTIFACT_DIR, next_version_label, set_active_version

logger = get_logger(__name__)

#: 低于该样本量不做留出集划分
MIN_SAMPLES_FOR_HOLDOUT = 15
#: Ridge 正则强度候选（交叉验证选择）
ALPHA_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def _cross_val_r2(X: np.ndarray, y: np.ndarray, alpha: float, folds: int) -> float | None:
    """K 折交叉验证的 R²。"""
    n = len(y)
    folds = max(2, min(folds, n))
    if n < folds:
        return None

    indices = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    chunks = np.array_split(indices, folds)

    predictions = np.full(n, np.nan, dtype=np.float64)
    for fold in range(folds):
        test_index = chunks[fold]
        train_index = np.concatenate([chunks[other] for other in range(folds) if other != fold])
        if len(train_index) < 2 or len(test_index) == 0:
            continue
        head = PropertyHead(algo="ridge", alpha=alpha)
        try:
            head.fit(X[train_index], y[train_index])
            predictions[test_index] = head.predict(X[test_index])
        except Exception as exc:
            logger.warning("交叉验证第 %d 折失败: %s", fold + 1, exc)

    mask = ~np.isnan(predictions)
    if mask.sum() < 3:
        return None
    actual = y[mask]
    predicted = predictions[mask]
    variance = float(((actual - actual.mean()) ** 2).sum())
    if variance <= 0:
        return None
    r2 = 1 - float(((actual - predicted) ** 2).sum()) / variance
    return round(r2, 4)


def select_alpha(X: np.ndarray, y: np.ndarray, folds: int) -> tuple[float, float | None]:
    """用交叉验证选择 Ridge 的正则强度。"""
    best_alpha = 1.0
    best_score: float | None = None
    for alpha in ALPHA_GRID:
        score = _cross_val_r2(X, y, alpha, folds)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score, best_alpha = score, alpha
    return best_alpha, best_score


def train_property_head(
    session: Session,
    property_name: str,
    *,
    algo: str = "ridge",
    min_samples: int = 8,
    test_ratio: float = 0.25,
    cv_folds: int = 5,
    notes: str | None = None,
) -> dict[str, Any]:
    """训练一个属性头并注册版本。

    Returns:
        包含 ``version``（ORM 对象）、``metrics``、``notes`` 等的字典。
    """
    started = time.perf_counter()
    result_notes: list[str] = []
    warnings: list[str] = []

    dataset = build_dataset(session, property_name)
    if not dataset.is_usable:
        raise ValidationError(
            f"属性 {property_name} 的数据不足以训练属性头",
            detail={"dataset": dataset.describe()},
        )
    if dataset.n_samples < min_samples:
        raise ValidationError(
            f"属性 {property_name} 仅有 {dataset.n_samples} 条可用样本，"
            f"低于设定下限 {min_samples}。请继续积累数据或降低 min_samples。",
            detail={"dataset": dataset.describe()},
        )

    assert dataset.embeddings is not None and dataset.targets is not None
    X = dataset.embeddings.astype(np.float64)
    y = dataset.targets.astype(np.float64)
    warnings.extend(dataset.warnings)

    # ---------- 划分 ----------
    n = len(y)
    use_holdout = n >= MIN_SAMPLES_FOR_HOLDOUT and test_ratio > 0
    if use_holdout:
        rng = np.random.default_rng(42)
        indices = np.arange(n)
        rng.shuffle(indices)
        test_size = max(2, int(round(n * test_ratio)))
        test_index, train_index = indices[:test_size], indices[test_size:]
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        result_notes.append(
            f"已按 {1 - test_ratio:.0%}/{test_ratio:.0%} 划分训练/留出集"
            f"（训练 {len(y_train)}，留出 {len(y_test)}），随机种子固定为 42。"
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None
        result_notes.append(
            f"样本量 {n} < {MIN_SAMPLES_FOR_HOLDOUT}，**未划分留出集**："
            "此时留出集评估波动极大，仅报告交叉验证指标。建议数据量提升后再训练。"
        )

    # ---------- 选择正则强度 ----------
    alpha = 1.0
    cv_r2: float | None = None
    if algo == "ridge":
        alpha, cv_r2 = select_alpha(X_train, y_train, cv_folds)
        result_notes.append(f"交叉验证选择正则强度 alpha={alpha}（{cv_folds} 折，CV R²={cv_r2}）")

    # ---------- 训练 ----------
    head = PropertyHead(algo=algo, alpha=alpha)
    head.fit(X_train, y_train)

    # ---------- 评估 ----------
    if use_holdout and X_test is not None and y_test is not None:
        predictions = head.predict(X_test)
        metrics = regression_metrics(y_test, predictions)
        metrics.notes.append("指标来自留出集（未参与训练）")
    else:
        predictions = head.predict(X_train)
        metrics = regression_metrics(y_train, predictions)
        metrics.notes.append(
            "指标来自**训练集自身**（无留出集），会明显偏乐观，不可作为泛化能力依据"
        )
    metrics.n_samples = n
    metrics.n_features = dataset.n_features
    metrics.cv_r2 = cv_r2
    metrics.notes.extend(result_notes)

    # ---------- 保存 + 注册 ----------
    version_label = next_version_label(session, property_name)
    settings = get_settings()
    artifact_dir = settings.models_dir / ALGO_ARTIFACT_DIR / property_name
    artifact_path = artifact_dir / f"{version_label}.npz"
    head.metrics = metrics
    head.save(artifact_path)

    # 记录本次消费到的最大 ExperimentRecord.id，作为后续增量训练的水位线
    consumed_ids = [rid for group in dataset.record_ids for rid in group]
    watermark = max(consumed_ids) if consumed_ids else None

    record = ModelVersion(
        property_name=property_name,
        version=version_label,
        algo=algo,
        base_model=settings.esm_model,
        n_samples=n,
        n_features=dataset.n_features,
        metrics=metrics.to_dict(),
        artifact_path=str(artifact_path),
        status="ready",
        is_active=False,
        last_record_id=watermark,
        note=notes,
    )
    session.add(record)
    session.flush()

    set_active_version(session, record)

    elapsed = time.perf_counter() - started
    logger.info(
        "属性头训练完成：属性=%s 版本=%s 样本=%d 耗时=%.1fs",
        property_name,
        version_label,
        n,
        elapsed,
    )

    return {
        "version": record,
        "metrics": metrics.to_dict(),
        "feature_dim": dataset.n_features,
        "train_samples": len(y_train),
        "test_samples": int(len(y_test) if y_test is not None else 0),
        "warnings": warnings,
        "notes": result_notes,
        "elapsed_seconds": round(elapsed, 2),
        "alpha": alpha,
        "labels": dataset.labels,
    }


def load_active_head(session: Session, property_name: str) -> PropertyHead | None:
    """加载某属性当前的激活模型；不存在时返回 ``None``。"""
    from .registry import get_active_version

    record = get_active_version(session, property_name)
    if record is None or not record.artifact_path:
        return None
    try:
        return PropertyHead.load(record.artifact_path)
    except Exception as exc:
        logger.warning("加载属性头失败 %s: %s", record.artifact_path, exc)
        return None


def incremental_train(
    session: Session,
    property_name: str,
    *,
    min_new_samples: int = 1,
) -> dict[str, Any]:
    """对已有激活模型做增量训练（仅追加激活版本之后新增的实验记录）。"""
    from .registry import get_active_version

    active = get_active_version(session, property_name)
    if active is None:
        raise ValidationError(
            f"属性 {property_name} 尚无激活模型，请先执行全量训练",
            detail={"hint": "调用 train_property_head 或使用 POST /api/model/train"},
        )

    head = load_active_head(session, property_name)
    if head is None:
        raise ValidationError(f"属性 {property_name} 的激活模型文件缺失，请重新全量训练")

    # 取激活版本尚未消费的新记录。
    # 用 last_record_id 水位线而不是 created_at：SQLite 的 CURRENT_TIMESTAMP 只有
    # 秒级精度，同一秒内产生的记录用时间比较会被漏判（实测踩过这个坑）。
    watermark = active.last_record_id or 0
    new_records = (
        session.query(ExperimentRecord)
        .filter(
            ExperimentRecord.property_name == property_name,
            ExperimentRecord.id > watermark,
        )
        .order_by(ExperimentRecord.id)
        .all()
    )

    if len(new_records) < min_new_samples:
        raise ValidationError(
            f"自版本 {active.version}（水位线 record_id={watermark}）以来"
            f"没有足够的新增实验记录（{len(new_records)} < {min_new_samples}）",
            detail={"active_version": active.version, "watermark": watermark},
        )

    # 构建新样本
    from .datasets import _resolve_target_sequence  # 内部工具，复用序列还原逻辑

    from ..db.models import ProteinSequence

    sequence_ids = {record.sequence_id for record in new_records if record.sequence_id is not None}
    cache: dict[int, str | None] = {}
    if sequence_ids:
        for item in session.query(ProteinSequence).filter(ProteinSequence.id.in_(sequence_ids)).all():
            cache[item.id] = item.sequence

    from ..services.embedding.esm2 import embed_sequence

    new_sequences: list[str] = []
    new_targets: list[float] = []
    skipped: list[dict[str, Any]] = []
    for record in new_records:
        sequence, reason = _resolve_target_sequence(record, cache)
        if sequence is None:
            skipped.append({"record_id": record.id, "reason": reason})
            continue
        new_sequences.append(sequence)
        new_targets.append(float(record.measured_value))

    if not new_sequences:
        raise ValidationError("新增记录均无法还原序列，无法增量训练", detail={"skipped": skipped[:20]})

    X_new = np.vstack([embed_sequence(item).mean for item in new_sequences]).astype(np.float64)
    y_new = np.asarray(new_targets, dtype=np.float64)

    head.partial_fit(X_new, y_new)

    # 保存为新版本
    version_label = next_version_label(session, property_name)
    settings = get_settings()
    artifact_path = settings.models_dir / ALGO_ARTIFACT_DIR / property_name / f"{version_label}.npz"
    head.save(artifact_path)

    # 在全量累积数据上评估，而不是只用这 2 条新样本——
    # 后者会产出 -251 这类毫无意义且严重误导的 R²（实测踩过）。
    buffered = head.evaluate_buffer()
    metrics = buffered or regression_metrics(head.predict(X_new), y_new)
    metrics.n_samples = head.n_samples
    metrics.n_features = head.feature_dim
    metrics.notes.append(
        f"增量训练：在版本 {active.version} 基础上追加 {len(new_sequences)} 条新样本，"
        f"累计 {head.n_samples} 条。"
    )
    if buffered is not None:
        metrics.notes.append(
            f"指标来自评估缓冲区（最多最近 {500} 条历史样本，本次 {buffered.n_samples} 条），"
            "覆盖新旧数据，可作为趋势参考；但仍非独立留出集，泛化能力需用外部数据验证。"
        )
    else:
        metrics.notes.append("评估缓冲区不可用，指标仅基于新增样本，不具参考价值。")

    record = ModelVersion(
        property_name=property_name,
        version=version_label,
        algo=head.algo,
        base_model=settings.esm_model,
        n_samples=head.n_samples,
        n_features=head.feature_dim,
        metrics=metrics.to_dict(),
        artifact_path=str(artifact_path),
        status="ready",
        is_active=False,
        last_record_id=max(item.id for item in new_records),
        note=f"增量训练，基线版本 {active.version}",
    )
    session.add(record)
    session.flush()
    set_active_version(session, record)

    return {
        "version": record,
        "metrics": metrics.to_dict(),
        "new_samples": len(new_sequences),
        "total_samples": head.n_samples,
        "skipped": skipped,
        "base_version": active.version,
        "algorithm": head.algo,
        "incremental_equivalent_to_full_refit": head.algo == "ridge",
    }
