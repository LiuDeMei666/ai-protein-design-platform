"""性质预测的统一结果结构。

设计原则（对应需求文档"输出可解释性评分，便于实验人员优先筛选"）
--------------------------------------------------------------
* 每个指标都是 0-100 分：**分数越高越好**（风险类指标为"安全性得分"）。
* 每项必须携带 ``algorithm``（算法来源）与 ``evidence``（逐条依据），
  禁止只给一个数字。
* ``confidence`` 显式表达置信度：纯规则计算为高置信，依赖模型零样本打分的为中置信。
* ``locus`` 保存位点级明细，直接供前端"位点轨道图"渲染。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ...core.config import load_platform_config

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
RISK_LABELS = {RISK_LOW: "低风险", RISK_MEDIUM: "中风险", RISK_HIGH: "高风险"}

#: 指标键 -> 中文标签（前端、导出与文档共用同一份定义，避免多处维护走样）
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


def metric_label(key: str) -> str:
    """取指标的中文标签。"""
    return METRIC_LABELS.get(key, key)


def _thresholds() -> tuple[float, float]:
    config = load_platform_config().get("risk_thresholds", {})
    return float(config.get("low", 0.33)), float(config.get("medium", 0.66))


def risk_from_score(score: float) -> str:
    """把 0-100 的"越高越好"分数映射为风险等级。"""
    low_cut, medium_cut = _thresholds()
    normalized = max(0.0, min(1.0, score / 100.0))
    if normalized >= medium_cut:
        return RISK_LOW
    if normalized >= low_cut:
        return RISK_MEDIUM
    return RISK_HIGH


@dataclass
class Evidence:
    """一条支撑证据。"""

    label: str
    value: float | str
    contribution: float = 0.0
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Metric:
    """单个性质指标的完整结果。"""

    key: str
    label: str
    score: float
    unit: str = ""
    value: float | None = None
    algorithm: str = ""
    rationale: str = ""
    confidence: float = 0.5
    evidence: list[Evidence] = field(default_factory=list)
    locus: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def risk(self) -> str:
        return risk_from_score(self.score)

    @property
    def risk_label(self) -> str:
        return RISK_LABELS[self.risk]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["risk"] = self.risk
        payload["risk_label"] = self.risk_label
        payload["evidence"] = [item.to_dict() for item in self.evidence]
        return payload


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """截断到区间内。"""
    return max(low, min(high, value))


def logistic(value: float, midpoint: float, steepness: float = 1.0) -> float:
    """S 形映射到 0-100，用于把无界物理量转成可比较的分数。"""
    import math

    return 100.0 / (1.0 + math.exp(-steepness * (value - midpoint)))


def linear_score(value: float, worst: float, best: float) -> float:
    """线性映射到 0-100；``worst`` 对应 0 分，``best`` 对应 100 分。

    ``best`` 可以小于 ``worst``（表示"越小越好"，例如不稳定指数）。
    """
    if best == worst:
        return 50.0
    ratio = (value - worst) / (best - worst)
    return clamp(ratio * 100.0)


def peak_score(value: float, optimum: float, tolerance: float) -> float:
    """"存在最优值"型映射：偏离最优越远分越低（如脯氨酸含量过高过低都不好）。"""
    if tolerance <= 0:
        return 50.0
    return clamp(100.0 * (1.0 - abs(value - optimum) / tolerance))


def aggregate_weighted(
    parts: list[tuple[float, float | None, Evidence]],
) -> tuple[float, list[Evidence]]:
    """加权聚合，并把每条证据的贡献折算为"对总分的实际贡献"。

    这是可解释性的关键：前端瀑布图直接消费 ``contribution``，
    各项贡献之和恰好等于总分，不存在无法归因的残差。

    Args:
        parts: ``[(权重, 子分数或 None, 证据), ...]``；子分数为 ``None`` 表示该证据
            当前不可用（例如没有结构时无 pLDDT），将从权重分母中剔除并重新归一。

    Returns:
        ``(总分, 证据列表)``
    """
    available = [(weight, score, item) for weight, score, item in parts if score is not None]
    total_weight = sum(weight for weight, _, _ in available)
    if total_weight <= 0:
        return 50.0, [item for _, _, item in parts]

    total = 0.0
    for weight, score, item in available:
        contribution = weight * float(score) / total_weight
        item.contribution = round(contribution, 2)
        total += contribution

    unavailable = [(weight, score, item) for weight, score, item in parts if score is None]
    for _, _, item in unavailable:
        item.contribution = 0.0
        if item.rationale and "不可用" not in item.rationale:
            item.rationale = f"{item.rationale}（该项因数据缺失未计入总分）"

    return clamp(total), [item for _, _, item in parts]


def build_summary(metrics: dict[str, Metric]) -> dict[str, Any]:
    """汇总为前端雷达图所需的紧凑结构。"""
    radar_keys = [
        "thermostability",
        "alkali_stability",
        "acid_stability",
        "solubility",
        "aggregation",
        "expression",
        "ptm_sites",
        "immunogenicity",
        "protease_resistance",
    ]
    radar = [
        {"key": key, "label": metrics[key].label, "score": round(metrics[key].score, 1)}
        for key in radar_keys
        if key in metrics
    ]
    risk_counts = {RISK_LOW: 0, RISK_MEDIUM: 0, RISK_HIGH: 0}
    for metric in metrics.values():
        risk_counts[metric.risk] += 1

    weakest = sorted(metrics.values(), key=lambda item: item.score)[:3]
    return {
        "radar": radar,
        "risk_counts": risk_counts,
        "overall_score": round(
            sum(metric.score for metric in metrics.values()) / max(1, len(metrics)), 1
        ),
        "weakest_metrics": [
            {"key": item.key, "label": item.label, "score": round(item.score, 1), "risk": item.risk}
            for item in weakest
        ],
    }
