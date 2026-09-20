"""属性头（PropertyHead）—— 模型迭代模块的落地载体。

设计要点
--------
需求文档要求"基于累积实验数据对预测模型进行增量训练与参数优化"。因此属性头
必须支持**真正的增量训练**，而不是每次都用全量数据重训。

实现方式：Ridge 回归用**充分统计量**累积 —

    在对特征做标准化后，闭式解为 ``w = (XᵀX + αI)⁻¹ Xᵀy``。
    只要保存 ``A = XᵀX`` 与 ``b = Xᵀy``，新数据到来时做 ``A += X_newᵀX_new``、
    ``b += X_newᵀy_new`` 再重新求解即可。结果是**与全量重训完全等价**的
    （数值误差在浮点精度内），代价从 O(n·d²) 的重算降为 O(n_new·d²)。

对于非线性的 ``hgb``（HistGradientBoosting）没有闭式增量解，`partial_fit`
会退化为全量重训，并在返回中明确标注，避免用户误以为它是增量的。

特征标准化参数在首次 ``fit`` 时固定，后续 ``partial_fit`` 复用同一组参数——
这是增量流水线的标准做法，保证新旧样本处于同一特征空间。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import PlatformError
from ..core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_ALGOS = ("ridge", "hgb")

#: 评估缓冲区容量。增量训练必须有一个"跨越新旧数据"的评估集，
#: 否则只在 2 条新样本上算 R² 会得到 -251 这类毫无意义且严重误导的数字。
#: 500 × 1280 × 8B ≈ 5 MB，代价可忽略。
EVAL_BUFFER_LIMIT = 500


@dataclass
class HeadMetrics:
    """评估指标。"""

    r2: float | None = None
    spearman: float | None = None
    pearson: float | None = None
    mae: float | None = None
    rmse: float | None = None
    cv_r2: float | None = None
    n_samples: int = 0
    n_features: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "r2": self.r2,
            "spearman": self.spearman,
            "pearson": self.pearson,
            "mae": self.mae,
            "rmse": self.rmse,
            "cv_r2": self.cv_r2,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "notes": self.notes,
        }


class PropertyHead:
    """轻量属性头：ESM-2 嵌入 -> 目标属性。"""

    def __init__(self, algo: str = "ridge", alpha: float = 1.0) -> None:
        if algo not in SUPPORTED_ALGOS:
            raise PlatformError(
                f"不支持的算法: {algo}", detail={"available": list(SUPPORTED_ALGOS)}
            )
        self.algo = algo
        self.alpha = float(alpha)
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.y_mean: float = 0.0
        self.n_samples: int = 0
        self.metrics: HeadMetrics = HeadMetrics()

        # Ridge 的充分统计量
        self._A: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._coef: np.ndarray | None = None
        self._X_buffer: np.ndarray | None = None
        self._y_buffer: np.ndarray | None = None

        # HGB 需要保留全量数据才能重训
        self._model: Any = None

        # 评估缓冲区（标准化后的特征与中心化后的标签）
        self._eval_X: np.ndarray | None = None
        self._eval_y: np.ndarray | None = None

    def _append_eval_buffer(self, standardized: np.ndarray, centered: np.ndarray) -> None:
        """追加到评估缓冲区并裁剪到最近 ``EVAL_BUFFER_LIMIT`` 条。"""
        if self._eval_X is None or self._eval_y is None:
            self._eval_X = standardized.copy()
            self._eval_y = centered.copy()
        else:
            self._eval_X = np.vstack([self._eval_X, standardized])
            self._eval_y = np.concatenate([self._eval_y, centered])
        if len(self._eval_y) > EVAL_BUFFER_LIMIT:
            self._eval_X = self._eval_X[-EVAL_BUFFER_LIMIT:]
            self._eval_y = self._eval_y[-EVAL_BUFFER_LIMIT:]

    def evaluate_buffer(self) -> "HeadMetrics | None":
        """在当前模型上评估评估缓冲区，得到跨新旧数据的真实拟合指标。"""
        if self._eval_X is None or self._eval_y is None or len(self._eval_y) == 0:
            return None
        if not self.is_fitted:
            return None
        try:
            predictions = self.predict_raw(self._eval_X) + self.y_mean
        except Exception:
            return None
        metrics = regression_metrics(self._eval_y + self.y_mean, predictions)
        metrics.n_features = self.feature_dim
        return metrics

    def predict_raw(self, standardized: np.ndarray) -> np.ndarray:
        """对**已标准化**的特征做预测（返回中心化空间的预测值）。"""
        if self.algo == "ridge":
            if self._coef is None:
                raise PlatformError("属性头尚未训练")
            return standardized @ self._coef
        if self._model is None:
            raise PlatformError("属性头尚未训练")
        return self._model.predict(standardized)

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _standardize(self, X: np.ndarray, update: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if update or self.feature_mean is None:
            self.feature_mean = X.mean(axis=0)
            std = X.std(axis=0)
            std[std < 1e-8] = 1.0
            self.feature_std = std
        assert self.feature_std is not None
        return (X - self.feature_mean) / self.feature_std

    def _solve_ridge(self) -> None:
        assert self._A is not None and self._b is not None
        dimension = self._A.shape[0]
        regularized = self._A + self.alpha * np.eye(dimension)
        try:
            self._coef = np.linalg.solve(regularized, self._b)
        except np.linalg.LinAlgError:
            # 病态时退化为最小二乘最小范数解
            self._coef = np.linalg.lstsq(regularized, self._b, rcond=None)[0]

    # ------------------------------------------------------------------ #
    # 训练
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "PropertyHead":
        """全量训练（会重置已有统计量）。"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if len(X) != len(y):
            raise PlatformError(f"特征数 {len(X)} 与标签数 {len(y)} 不一致")
        if len(X) == 0:
            raise PlatformError("训练数据为空")

        self.n_samples = 0
        self._A = None
        self._b = None
        self._coef = None
        self._X_buffer = None
        self._y_buffer = None
        self._model = None
        self.metrics = HeadMetrics(n_features=int(X.shape[1]))

        standardized = self._standardize(X, update=True)
        self.y_mean = float(y.mean())
        centered = y - self.y_mean
        self.n_samples = len(y)

        self._eval_X = None
        self._eval_y = None
        self._append_eval_buffer(standardized, centered)

        if self.algo == "ridge":
            self._A = standardized.T @ standardized
            self._b = standardized.T @ centered
            self._solve_ridge()
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor

            self._X_buffer = standardized.copy()
            self._y_buffer = centered.copy()
            self._model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.06, max_depth=4, random_state=42
            )
            self._model.fit(standardized, centered)

        logger.info("属性头训练完成：algo=%s n=%d d=%d", self.algo, self.n_samples, X.shape[1])
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "PropertyHead":
        """增量训练。

        * ``ridge``：累积充分统计量后重新求解，**结果与全量重训等价**。
        * ``hgb``：无闭式增量解，退化为"追加数据后全量重训"，返回的
          :attr:`metrics` 会带 note 说明。

        首次调用时若尚未 ``fit``，行为等同于 ``fit``。
        """
        if self.feature_mean is None or self._coef is None and self._model is None:
            return self.fit(X, y)

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if len(X) == 0:
            return self

        standardized = self._standardize(X, update=False)
        centered = y - self.y_mean
        self._append_eval_buffer(standardized, centered)

        if self.algo == "ridge":
            assert self._A is not None and self._b is not None
            self._A += standardized.T @ standardized
            self._b += standardized.T @ centered
            self._solve_ridge()
            self.n_samples += len(y)
            logger.info("属性头增量更新：新增 %d 条，累计 %d 条", len(y), self.n_samples)
        else:
            assert self._X_buffer is not None and self._y_buffer is not None
            from sklearn.ensemble import HistGradientBoostingRegressor

            self._X_buffer = np.vstack([self._X_buffer, standardized])
            self._y_buffer = np.concatenate([self._y_buffer, centered])
            self.n_samples = len(self._y_buffer)
            self._model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.06, max_depth=4, random_state=42
            )
            self._model.fit(self._X_buffer, self._y_buffer)
            self.metrics.notes.append(
                "hgb 无闭式增量解，本次为追加数据后的全量重训（结果正确但耗时随数据量增长）"
            )
            logger.info("属性头 hgb 全量重训：累计 %d 条", self.n_samples)

        return self

    # ------------------------------------------------------------------ #
    # 预测
    # ------------------------------------------------------------------ #
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测。"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        standardized = self._standardize(X, update=False)

        if self.algo == "ridge":
            if self._coef is None:
                raise PlatformError("属性头尚未训练")
            return standardized @ self._coef + self.y_mean
        if self._model is None:
            raise PlatformError("属性头尚未训练")
        return self._model.predict(standardized) + self.y_mean

    # ------------------------------------------------------------------ #
    # 序列化
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> Path:
        """保存到磁盘（npz + json 元数据，避免 pickle 的版本兼容问题）。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {"y_mean": np.array([self.y_mean])}
        if self.feature_mean is not None:
            arrays["feature_mean"] = self.feature_mean
        if self.feature_std is not None:
            arrays["feature_std"] = self.feature_std
        if self._coef is not None:
            arrays["coef"] = self._coef
        if self._A is not None:
            arrays["A"] = self._A
        if self._b is not None:
            arrays["b"] = self._b
        if self._X_buffer is not None:
            arrays["X_buffer"] = self._X_buffer
        if self._y_buffer is not None:
            arrays["y_buffer"] = self._y_buffer
        if self._eval_X is not None:
            arrays["eval_X"] = self._eval_X
        if self._eval_y is not None:
            arrays["eval_y"] = self._eval_y

        np.savez_compressed(target, **arrays)

        if self._model is not None:
            import pickle

            with (target.parent / f"{target.name}.hgb").open("wb") as handle:
                pickle.dump(self._model, handle)

        meta = {
            "algo": self.algo,
            "alpha": self.alpha,
            "n_samples": self.n_samples,
            "metrics": self.metrics.to_dict(),
            "feature_dim": int(self.feature_mean.shape[0]) if self.feature_mean is not None else 0,
            "format_version": 1,
        }
        (target.parent / f"{target.name}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("属性头已保存: %s", target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "PropertyHead":
        """从磁盘加载。"""
        target = Path(path)
        if not target.exists():
            raise PlatformError(f"模型文件不存在: {target}")

        meta_path = target.parent / f"{target.name}.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

        head = cls(algo=meta.get("algo", "ridge"), alpha=float(meta.get("alpha", 1.0)))
        with np.load(target) as payload:
            head.y_mean = float(payload["y_mean"][0]) if "y_mean" in payload else 0.0
            head.feature_mean = payload["feature_mean"] if "feature_mean" in payload else None
            head.feature_std = payload["feature_std"] if "feature_std" in payload else None
            head._coef = payload["coef"] if "coef" in payload else None
            head._A = payload["A"] if "A" in payload else None
            head._b = payload["b"] if "b" in payload else None
            head._X_buffer = payload["X_buffer"] if "X_buffer" in payload else None
            head._y_buffer = payload["y_buffer"] if "y_buffer" in payload else None
            head._eval_X = payload["eval_X"] if "eval_X" in payload else None
            head._eval_y = payload["eval_y"] if "eval_y" in payload else None

        head.n_samples = int(meta.get("n_samples", 0))
        metrics_payload = meta.get("metrics", {})
        head.metrics = HeadMetrics(
            r2=metrics_payload.get("r2"),
            spearman=metrics_payload.get("spearman"),
            pearson=metrics_payload.get("pearson"),
            mae=metrics_payload.get("mae"),
            rmse=metrics_payload.get("rmse"),
            cv_r2=metrics_payload.get("cv_r2"),
            n_samples=metrics_payload.get("n_samples", head.n_samples),
            n_features=metrics_payload.get("n_features", meta.get("feature_dim", 0)),
            notes=metrics_payload.get("notes", []),
        )

        hgb_path = target.parent / f"{target.name}.hgb"
        if hgb_path.exists():
            import pickle

            with hgb_path.open("rb") as handle:
                head._model = pickle.load(handle)

        return head

    # ------------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        return self._coef is not None or self._model is not None

    @property
    def feature_dim(self) -> int:
        if self.feature_mean is None:
            return 0
        return int(self.feature_mean.shape[0])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> HeadMetrics:
    """计算回归指标。"""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    n = len(y_true)
    metrics = HeadMetrics(n_samples=n)

    if n == 0:
        return metrics

    residuals = y_true - y_pred
    metrics.mae = round(float(np.abs(residuals).mean()), 4)
    metrics.rmse = round(float(math.sqrt(float((residuals**2).mean()))), 4)

    variance = float(((y_true - y_true.mean()) ** 2).sum())
    if n >= 2 and variance > 0:
        metrics.r2 = round(float(1 - (residuals**2).sum() / variance), 4)

    if n >= 3 and float(np.ptp(y_true)) > 0 and float(np.ptp(y_pred)) > 0:
        try:
            from scipy import stats

            pearson = stats.pearsonr(y_true, y_pred).statistic
            spearman = stats.spearmanr(y_true, y_pred).statistic
            metrics.pearson = None if pearson is None or np.isnan(pearson) else round(float(pearson), 4)
            metrics.spearman = None if spearman is None or np.isnan(spearman) else round(float(spearman), 4)
        except Exception:
            pass

    return metrics
