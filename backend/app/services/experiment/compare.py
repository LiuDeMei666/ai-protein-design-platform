"""预测值与实测值的对比分析。

方法学说明（关系到结论可信度，务必理解）
----------------------------------------
平台输出的性质指标是 **0-100 的评分**，而实验实测值有各自量纲（°C、mg/L、U/mg…）。
两者**不在同一尺度**上，因此不能直接计算 MAE/RMSE。正确的做法是：

1. **秩相关（Spearman ρ）** —— 这是零样本预测器评估的标准做法（ProteinGym 等
   基准均以 Spearman 为主指标），衡量"排序是否一致"，不要求同尺度。
2. **皮尔逊相关（Pearson r）** —— 衡量线性相关强度。
3. **线性校准后的误差** —— 在数据上拟合 ``实测值 = a × 评分 + b``，再用校准后的
   预测计算 MAE/RMSE/R²，单位与实测一致，便于实验人员直观理解偏差幅度。

**重要局限**：校准系数是在同一批数据上拟合的（in-sample），样本量小时 R² 会
系统性偏高。报告中会显式标注样本量与该校准方式，并给出"样本量不足，仅供参考"的
提示。企业数据积累到 30+ 条后应改用留一交叉验证。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...core.errors import ValidationError
from ...core.logging import get_logger
from ..property.pipeline import predict_properties

logger = get_logger(__name__)

#: 平台性质指标键 -> 中文标签
METRIC_LABELS: dict[str, str] = {
    "thermostability": "热稳定性",
    "alkali_stability": "碱稳定性",
    "acid_stability": "酸稳定性",
    "solubility": "溶解性",
    "aggregation": "聚集风险",
    "expression": "表达量趋势",
    "ptm_sites": "修饰位点风险",
    "immunogenicity": "免疫原性风险",
    "protease_resistance": "宿主蛋白酶抗性",
}

#: 实测属性名 -> 平台指标键（对比时使用；与 ingest 的别名表保持一致）
COMPARABLE_PROPERTIES: dict[str, str] = {
    "thermostability": "thermostability",
    "alkali_stability": "alkali_stability",
    "acid_stability": "acid_stability",
    "solubility": "solubility",
    "aggregation": "aggregation",
    "expression": "expression",
    "ptm_sites": "ptm_sites",
    "immunogenicity": "immunogenicity",
    "protease_resistance": "protease_resistance",
}

#: 低于此样本量时不给出定量结论
MIN_SAMPLES_FOR_STATS = 4

_MUTATION_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")


@dataclass
class ComparisonOptions:
    """对比分析选项。"""

    use_esm: bool = False
    host_system: str = "ecoli"
    protein_type: str = "generic"
    structure_stats: dict[str, Any] | None = None


@dataclass
class PredictionCache:
    """按突变标签缓存预测结果，避免重复计算。"""

    sequence: str
    options: ComparisonOptions
    cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get(self, mutation_label: str) -> dict[str, Any]:
        key = mutation_label or "__wild_type__"
        if key not in self.cache:
            self.cache[key] = self._predict(mutation_label)
        return self.cache[key]

    def _predict(self, mutation_label: str) -> dict[str, Any]:
        try:
            target = apply_mutations(self.sequence, mutation_label) if mutation_label else self.sequence
        except ValueError as exc:
            return {"error": str(exc), "metrics": {}}

        report = predict_properties(
            target,
            protein_type=self.options.protein_type,
            host_system=self.options.host_system,
            structure_stats=None if mutation_label else self.options.structure_stats,
            use_esm=self.options.use_esm,
        )
        return {
            "metrics": {key: metric.score for key, metric in report.metrics.items()},
            "detail": {key: metric.to_dict() for key, metric in report.metrics.items()},
            "sequence": target,
            "warnings": report.warnings,
        }


def parse_mutation_label(label: str, sequence: str) -> dict[str, Any]:
    """解析突变标签 ``A123V``。"""
    text = (label or "").strip().upper()
    match = _MUTATION_PATTERN.match(text)
    if match is None:
        raise ValueError(f"突变标签格式不合法：{label!r}，应为 野生型残基+位置+突变残基（如 A123V）")
    wild_type, position_text, mutant = match.groups()
    position = int(position_text)
    if not (1 <= position <= len(sequence)):
        raise ValueError(f"突变位置 {position} 超出序列长度 {len(sequence)}")
    actual = sequence[position - 1]
    if actual != wild_type:
        raise ValueError(
            f"突变标签 {text} 与序列不符：第 {position} 位实际为 {actual}，标签写的是 {wild_type}"
        )
    return {"position": position, "wild_type": wild_type, "mutant": mutant, "label": text}


def apply_mutations(sequence: str, label: str) -> str:
    """把突变标签应用到序列上（支持逗号分隔的多点突变）。"""
    if not label:
        return sequence
    chars = list(sequence)
    seen: set[int] = set()
    for part in str(label).split(","):
        part = part.strip()
        if not part:
            continue
        info = parse_mutation_label(part, sequence)
        index = info["position"] - 1
        if index in seen:
            raise ValueError(f"突变位置重复：{info['position']}")
        seen.add(index)
        chars[index] = info["mutant"]
    return "".join(chars)


def _calibrate(predicted: np.ndarray, measured: np.ndarray) -> tuple[float, float, np.ndarray]:
    """一阶线性校准：``measured ≈ a * predicted + b``。"""
    if len(predicted) < 2 or float(np.ptp(predicted)) == 0.0:
        return 0.0, float(measured.mean()) if len(measured) else 0.0, np.full_like(predicted, measured.mean() if len(measured) else 0.0)
    slope, intercept = np.polyfit(predicted, measured, 1)
    return float(slope), float(intercept), slope * predicted + intercept


def _safe_corr(first: np.ndarray, second: np.ndarray, method: str) -> float | None:
    if len(first) < 3:
        return None
    if float(np.ptp(first)) == 0.0 or float(np.ptp(second)) == 0.0:
        return None
    try:
        from scipy import stats

        if method == "pearson":
            value = stats.pearsonr(first, second).statistic
        else:
            value = stats.spearmanr(first, second).statistic
        if value is None or np.isnan(value):
            return None
        return round(float(value), 4)
    except Exception:
        return None


def _verdict(n_pairs: int, spearman: float | None) -> str:
    if n_pairs < MIN_SAMPLES_FOR_STATS:
        return f"样本量不足（{n_pairs} 条 < {MIN_SAMPLES_FOR_STATS}），仅作参考，不支持定量结论"
    if spearman is None:
        return "预测值与实测值均无变化或数据异常，无法评估相关性"
    magnitude = abs(spearman)
    if magnitude >= 0.6:
        return "排序一致性良好：模型可用于候选筛选（Spearman |ρ| ≥ 0.6）"
    if magnitude >= 0.3:
        return "排序一致性中等：可用于粗筛，建议结合人工判断"
    return "排序一致性偏弱：模型对该属性的预测能力不足，建议扩充实验数据后重新训练属性头"


def compare_records(
    sequence: str,
    records: list[dict[str, Any]],
    options: ComparisonOptions | None = None,
) -> dict[str, Any]:
    """执行预测-实测对比。

    Args:
        sequence: 野生型序列。
        records: 实验记录列表，每条需含 ``mutation`` 与 ``property_name``、``measured_value``。
        options: 对比选项。
    """
    options = options or ComparisonOptions()
    cache = PredictionCache(sequence=sequence, options=options)

    # 按属性聚合配对
    grouped: dict[str, list[dict[str, Any]]] = {}
    unmatched: list[dict[str, Any]] = []
    warnings: list[str] = []

    for index, record in enumerate(records):
        property_name = str(record.get("property_name") or "").strip()
        metric_key = COMPARABLE_PROPERTIES.get(property_name)
        if metric_key is None:
            unmatched.append(
                {
                    "index": index,
                    "mutation": record.get("mutation", ""),
                    "property_name": property_name,
                    "reason": "该属性不在平台可预测范围内，无法对比",
                }
            )
            continue

        value = record.get("measured_value")
        if value is None:
            unmatched.append(
                {
                    "index": index,
                    "mutation": record.get("mutation", ""),
                    "property_name": property_name,
                    "reason": "实测值为空",
                }
            )
            continue

        prediction = cache.get(str(record.get("mutation") or ""))
        if prediction.get("error"):
            unmatched.append(
                {
                    "index": index,
                    "mutation": record.get("mutation", ""),
                    "property_name": property_name,
                    "reason": prediction["error"],
                }
            )
            continue

        score = prediction["metrics"].get(metric_key)
        if score is None:
            unmatched.append(
                {
                    "index": index,
                    "mutation": record.get("mutation", ""),
                    "property_name": property_name,
                    "reason": "预测结果中缺少该指标",
                }
            )
            continue

        grouped.setdefault(metric_key, []).append(
            {
                "mutation": record.get("mutation") or "野生型",
                "predicted": float(score),
                "measured": float(value),
                "unit": record.get("unit"),
                "condition": record.get("condition"),
                "replicate": record.get("replicate"),
            }
        )

    # 逐属性统计
    property_reports: list[dict[str, Any]] = []
    for metric_key, pairs in grouped.items():
        predicted = np.array([item["predicted"] for item in pairs], dtype=np.float64)
        measured = np.array([item["measured"] for item in pairs], dtype=np.float64)

        pearson = _safe_corr(predicted, measured, "pearson")
        spearman = _safe_corr(predicted, measured, "spearman")
        slope, intercept, calibrated = _calibrate(predicted, measured)

        residuals = measured - calibrated
        mae = float(np.abs(residuals).mean()) if len(residuals) else None
        rmse = float(np.sqrt((residuals**2).mean())) if len(residuals) else None
        total_variance = float(((measured - measured.mean()) ** 2).sum())
        r2 = (
            float(1 - (residuals**2).sum() / total_variance)
            if total_variance > 0 and len(measured) >= 2
            else None
        )
        bias = float(residuals.mean()) if len(residuals) else None

        points = [
            {
                "mutation": item["mutation"],
                "predicted_score": round(item["predicted"], 2),
                "measured_value": item["measured"],
                "calibrated_prediction": round(float(calibrated[index]), 3),
                "residual": round(float(residuals[index]), 3),
                "unit": item["unit"],
            }
            for index, item in enumerate(pairs)
        ]
        worst = sorted(points, key=lambda item: -abs(item["residual"]))[:5]

        property_reports.append(
            {
                "property_name": metric_key,
                "label": METRIC_LABELS.get(metric_key, metric_key),
                "n_pairs": len(pairs),
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "mae": round(mae, 4) if mae is not None else None,
                "rmse": round(rmse, 4) if rmse is not None else None,
                "r2": round(r2, 4) if r2 is not None else None,
                "bias": round(bias, 4) if bias is not None else None,
                "unit": next((item["unit"] for item in pairs if item["unit"]), None),
                "verdict": _verdict(len(pairs), spearman),
                "points": points,
                "worst_offsets": worst,
                "calibration": {
                    "slope": round(slope, 6),
                    "intercept": round(intercept, 4),
                    "method": "in-sample 一阶线性校准（样本量 ≥ 30 时建议改用留一交叉验证）",
                },
            }
        )

    property_reports.sort(key=lambda item: -item["n_pairs"])
    total_pairs = sum(item["n_pairs"] for item in property_reports)

    if not property_reports:
        overall = "没有任何可配对的记录：请确认属性名与平台的指标键一致，且突变标签与序列匹配。"
    elif total_pairs < MIN_SAMPLES_FOR_STATS:
        overall = (
            f"仅 {total_pairs} 条可配对记录，样本量不足以评估模型可靠性。"
            "建议每个属性至少积累 10 条以上实测数据。"
        )
    else:
        rhos = [item["spearman_rho"] for item in property_reports if item["spearman_rho"] is not None]
        if not rhos:
            overall = "配对成功但预测值无差异，无法计算相关性。"
        else:
            mean_rho = float(np.mean(rhos))
            overall = (
                f"共 {total_pairs} 条配对记录，跨 {len(property_reports)} 个属性，"
                f"平均 Spearman ρ = {mean_rho:.3f}。"
                + ("整体排序一致性良好。" if abs(mean_rho) >= 0.5 else "整体一致性一般，模型仍有提升空间。")
            )

    recommendations: list[str] = []
    low = [item for item in property_reports if item["spearman_rho"] is not None and abs(item["spearman_rho"]) < 0.3]
    if low:
        recommendations.append(
            "以下属性的预测与实测排序一致性偏弱，建议优先用这些数据训练属性头："
            + "、".join(item["label"] for item in low)
        )
    rich = [item for item in property_reports if item["n_pairs"] >= 10]
    if rich:
        recommendations.append(
            "以下属性已有 ≥10 条配对数据，具备训练属性头的条件："
            + "、".join(item["label"] for item in rich)
        )
    else:
        recommendations.append("尚无任何属性达到 10 条配对数据，建议继续积累后再触发增量训练。")
    if unmatched:
        recommendations.append(
            f"有 {len(unmatched)} 条记录未能配对（属性不可预测 / 突变标签与序列不符），"
            "请在导入预览中逐条核对。"
        )

    return {
        "total_records": len(records),
        "matched_pairs": total_pairs,
        "unmatched_records": len(unmatched),
        "unmatched_detail": unmatched[:50],
        "properties": property_reports,
        "overall_verdict": overall,
        "recommendations": recommendations,
        "warnings": warnings,
        "methodology": {
            "primary_metric": "Spearman ρ（秩相关，不要求同尺度，是零样本预测器的标准评估指标）",
            "error_metrics": "MAE/RMSE/R² 基于一阶线性校准后的预测值，单位与实测一致",
            "caveat": "校准系数在样本内拟合，样本量小时 R² 会偏高；样本量 ≥ 30 时请改用交叉验证",
            "min_samples": MIN_SAMPLES_FOR_STATS,
        },
    }
