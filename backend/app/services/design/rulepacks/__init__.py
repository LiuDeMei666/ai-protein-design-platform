"""规则包注册表。"""

from __future__ import annotations

from typing import Any

from ....core.config import load_platform_config
from ....core.errors import ValidationError
from ....core.logging import get_logger
from .base import GenericRulePack, RuleContext, RulePack, RuleVerdict
from .collagen import CollagenRulePack
from .protease import ProteaseRulePack
from .protein_a import ProteinARulePack

logger = get_logger(__name__)

__all__ = [
    "RulePack",
    "RuleContext",
    "RuleVerdict",
    "GenericRulePack",
    "CollagenRulePack",
    "ProteaseRulePack",
    "ProteinARulePack",
    "get_rulepack",
    "rulepack_catalog",
    "PROTEIN_TYPES",
]

PROTEIN_TYPES: tuple[str, ...] = ("collagen", "protease", "protein_a", "generic")

_RULEPACKS: dict[str, RulePack] = {}


def get_rulepack(category: str) -> RulePack:
    """按蛋白类型取规则包（进程内缓存）。

    ``generic`` 规则包始终叠加在专用规则包之上：专用规则包先给出结论，
    再由通用规则补充"表面优先/避免游离 Cys"等普适约束——
    两者取更严格的一方（implemented in :func:`merge_verdicts`）。
    """
    category = (category or "generic").strip().lower()
    if category not in PROTEIN_TYPES:
        raise ValidationError(
            f"未知的蛋白类型: {category}",
            detail={"available": list(PROTEIN_TYPES)},
        )
    if category not in _RULEPACKS:
        config = load_platform_config().get("rulepacks", {})
        if category == "collagen":
            settings = config.get("collagen", {})
            _RULEPACKS[category] = CollagenRulePack(
                gly_strict=bool(settings.get("gly_strict", True)),
                protect_non_product_region=bool(
                    settings.get("protect_non_product_region", True)
                ),
            )
        elif category == "protease":
            _RULEPACKS[category] = ProteaseRulePack()
        elif category == "protein_a":
            _RULEPACKS[category] = ProteinARulePack()
        else:
            settings = config.get("generic", {})
            _RULEPACKS[category] = GenericRulePack(
                avoid_free_cysteine=bool(settings.get("avoid_free_cysteine", True)),
                avoid_glycine_in_helix=bool(settings.get("avoid_glycine_in_helix", True)),
            )
    return _RULEPACKS[category]


def get_generic_rulepack() -> GenericRulePack:
    """通用规则包（始终参与）。"""
    if "generic" not in _RULEPACKS:
        _RULEPACKS["generic"] = GenericRulePack()
    return _RULEPACKS["generic"]  # type: ignore[return-value]


def merge_verdicts(primary: RuleVerdict, secondary: RuleVerdict) -> RuleVerdict:
    """合并两个规则结论：取更保守的因子，合并提示与证据。"""
    return RuleVerdict(
        factor=min(primary.factor, secondary.factor),
        note="；".join(item for item in (primary.note, secondary.note) if item),
        flags=list(dict.fromkeys([*primary.flags, *secondary.flags])),
        evidence={**secondary.evidence, **primary.evidence},
    )


def rulepack_catalog() -> dict[str, dict[str, Any]]:
    """全部规则包的元信息（供前端展示与文档生成）。"""
    return {category: get_rulepack(category).describe() for category in PROTEIN_TYPES}


def reset_rulepacks() -> None:
    """测试用：清空缓存。"""
    _RULEPACKS.clear()
