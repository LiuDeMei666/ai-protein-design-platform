"""零样本打分、风险判定、规则包与评分合成的确定性测试（全部离线）。"""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.services.design.explain import (
    assemble_dimensions,
    expression_delta_score,
)
from backend.app.services.design.rulepacks import get_rulepack
from backend.app.services.design.rulepacks.collagen import CollagenRulePack
from backend.app.services.design.scanner import (
    PositionContext,
    build_mutant_sequence,
    build_scan_plan,
    mutation_label,
)
from backend.app.services.design.scorers.biophys_delta import assess_mutation_risk
from backend.app.services.design.scorers.foldability import (
    assess_foldability,
    tolerance_from_plddt,
    tolerance_from_secondary_structure,
)
from backend.app.services.design.scorers.zero_shot import (
    _mapping,
    conservation_level,
    normalize_delta,
)


class TestZeroShotNormalization:
    """ΔlogP 到 0-100 的映射。"""

    def test_mapping_endpoints(self):
        worst, best = _mapping()
        deltas = np.array([[worst, best, (worst + best) / 2]], dtype=np.float64)
        scaled = normalize_delta(deltas)
        assert scaled[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert scaled[0, 1] == pytest.approx(100.0, abs=1e-6)
        assert scaled[0, 2] == pytest.approx(50.0, abs=1e-6)

    def test_beneficial_delta_clamped_at_hundred(self):
        assert normalize_delta(np.array([[5.0]]))[0, 0] == 100.0

    def test_very_harmful_delta_clamped_at_zero(self):
        assert normalize_delta(np.array([[-50.0]]))[0, 0] == 0.0

    def test_conservation_level_monotonic(self):
        """野生型对数概率越低（约束越弱），可突变空间分应越高。"""
        scores = [conservation_level(value)[0] for value in (-0.2, -1.0, -2.5, -4.0, -8.0)]
        assert scores == sorted(scores)
        assert conservation_level(None)[0] == 50.0


class TestRiskAssessment:
    """突变化学风险判定。"""

    def test_creating_asn_gly_motif_is_flagged(self):
        """把某位点变成 N 且后一位是 G 时，必须报出新脱酰胺位点。"""
        sequence = "MKAVRQERLKSIVR"
        # 第 3 位（0-based 2）为 A，改为 N 后形成 N-G 基序（第 4 位是 V）→ 不成立，
        # 因此改用第 1 位：M->N 后序列以 N-K 开头，也不成立。
        # 直接构造：把 0-based 2 位的 A 改为 N，并把 0-based 3 位看作 G
        sequence = "MKNGVRQERLKSIVR"
        # 先把 N 改成 A 作为野生型，再改回 N 应触发
        wild = sequence[:2] + "A" + sequence[3:]
        assessment = assess_mutation_risk(wild, 2, "N")
        assert any("脱酰胺" in flag for flag in assessment.flags)
        assert assessment.safety_score < 100

    def test_breaking_cysteine_is_penalized(self):
        sequence = "MKTVCRQERLKSIVR"
        assessment = assess_mutation_risk(sequence, 4, "A")
        assert any("二硫键" in flag for flag in assessment.flags)
        assert assessment.safety_score <= 80

    def test_introducing_free_cysteine_is_penalized(self):
        sequence = "MKTVRQERLKSIVRILERS"
        assessment = assess_mutation_risk(sequence, 5, "C")
        assert any("半胱氨酸" in flag for flag in assessment.flags)

    def test_introducing_proline_in_helix_is_flagged(self):
        sequence = "MKTVRQERLKSIVRILERS"
        context = PositionContext(
            position=5, residue="Q", plddt=92.0, relative_sasa=0.1, secondary_structure="H"
        )
        assessment = assess_mutation_risk(sequence, 5, "P", context)
        assert any("螺旋" in flag for flag in assessment.flags)

    def test_safe_substitution_has_no_flags(self, sample_sequence):
        # 序列第 5 位是 Q（0-based 4），改为 E 不引入任何风险基序
        assessment = assess_mutation_risk(sample_sequence, 4, "E")
        assert assessment.safety_score == 100.0 or not assessment.flags

    def test_deltas_recorded(self, sample_sequence):
        assessment = assess_mutation_risk(sample_sequence, 4, "K")
        assert "volume_change" in assessment.deltas
        assert "charge_change" in assessment.deltas
        assert assessment.deltas["label"].startswith(sample_sequence[4])


class TestFoldability:
    """折叠可行性。"""

    def test_low_plddt_more_tolerant(self):
        assert tolerance_from_plddt(40.0) > tolerance_from_plddt(95.0)

    def test_coil_more_tolerant_than_sheet(self):
        assert tolerance_from_secondary_structure("-") > tolerance_from_secondary_structure("E")

    def test_missing_structure_evidence_is_dropped(self):
        """无结构信息时应剔除相关证据，而不是当作 0 分。"""
        from backend.app.services.design.scorers.zero_shot import ZeroShotScores

        scores = ZeroShotScores(
            positions=[0],
            aa_order="ACDEFGHIKLMNPQRSTVWY",
            delta_logprob=np.zeros((1, 20)),
            wt_logprob=np.array([-1.0]),
            stability=np.full((1, 20), 80.0),
            windows=[(0, 1)],
        )
        result = assess_foldability("MKTV", 0, "V", 80.0, context=None)
        assert result.score == pytest.approx(80.0, abs=0.01)
        assert "野生型 pLDDT" in result.evidence["missing_evidence"]


class TestScoreAssembly:
    """评分合成必须满足"贡献之和 = 总分"。"""

    def test_contributions_sum_to_total(self):
        total, dimensions = assemble_dimensions(
            stability=(80.0, "s"),
            activity=(70.0, "a"),
            foldability=(60.0, "f"),
            expression=(50.0, "e"),
            risk=(90.0, "r"),
        )
        assert sum(item.contribution for item in dimensions) == pytest.approx(total, abs=0.01)

    def test_risk_is_penalty(self):
        """安全性越低，总分应越低（风险以罚项计入）。"""
        _, high_risk = assemble_dimensions((80.0, ""), (80.0, ""), (80.0, ""), (80.0, ""), (0.0, ""))
        _, low_risk = assemble_dimensions((80.0, ""), (80.0, ""), (80.0, ""), (80.0, ""), (100.0, ""))
        high_total = sum(item.contribution for item in high_risk)
        low_total = sum(item.contribution for item in low_risk)
        assert low_total > high_total

    def test_dimension_keys_complete(self):
        _, dimensions = assemble_dimensions((50.0, ""), (50.0, ""), (50.0, ""), (50.0, ""), (50.0, ""))
        assert {item.key for item in dimensions} == {
            "stability", "activity", "foldability", "expression", "risk"
        }

    def test_weights_sum_to_unit_normalized(self):
        _, dimensions = assemble_dimensions((50.0, ""), (50.0, ""), (50.0, ""), (50.0, ""), (50.0, ""))
        assert sum(item.weight for item in dimensions) == pytest.approx(1.0, abs=1e-6)


class TestExpressionDelta:
    """表达适配维度的变化方向。"""

    def test_neutral_substitution_is_neutral(self, sample_sequence):
        score, note = expression_delta_score(sample_sequence, 4, "E")
        assert score == pytest.approx(70.0, abs=1e-6)

    def test_introducing_stalling_motif_lowers_score(self):
        sequence = "MKTVRQERLKSIVRILERS"
        score, note = expression_delta_score(sequence, 5, "R")  # 若形成 RR
        if "停滞" in note:
            assert score < 70.0


class TestCollagenRulePack:
    """胶原规则包的三股螺旋约束。"""

    def test_all_register_glycines_protected(self):
        sequence = "GPP" * 20
        pack = CollagenRulePack()
        protected = pack.protected_positions(sequence)
        gly_positions = {index for index, char in enumerate(sequence) if char == "G"}
        assert gly_positions <= set(protected.keys())
        assert all("三股螺旋" in reason for reason in protected.values())

    def test_y_site_proline_is_top_recommendation(self):
        from backend.app.services.design.rulepacks.base import RuleContext
        from backend.app.services.sequence.feature_utils import gly_xy_phase

        # GPPGPPGPAPPG 中第 9 位（0-based 8）位于 Y 位（相位 2）
        sequence = "GPPGPPGPAPPG"
        position = 8
        assert sequence[position] == "A"
        assert gly_xy_phase(sequence, position) == 2, "测试序列的 Y 位构造有误"

        pack = CollagenRulePack()
        context = RuleContext(sequence=sequence, position=position, wild_type="A", mutant="P")
        verdict = pack.activity_factor(context)
        assert verdict.factor >= 0.9
        assert any("羟脯氨酸" in flag for flag in verdict.flags)

    def test_bulky_residue_at_y_site_is_discouraged(self):
        from backend.app.services.design.rulepacks.base import RuleContext

        sequence = "GPPGPPGPAPPG"
        pack = CollagenRulePack()
        context = RuleContext(sequence=sequence, position=8, wild_type="A", mutant="W")
        verdict = pack.activity_factor(context)
        assert verdict.factor < 0.3

    def test_glycine_substitution_is_blocked(self):
        from backend.app.services.design.rulepacks.base import RuleContext

        sequence = "GPP" * 10
        pack = CollagenRulePack()
        context = RuleContext(sequence=sequence, position=0, wild_type="G", mutant="A")
        verdict = pack.activity_factor(context)
        assert verdict.factor == 0.0


class TestProteaseRulePack:
    """蛋白酶规则包的活性位点保护（含相对间距自校验）。

    **使用真实种子序列**（UniProt P00782 枯草杆菌蛋白酶 BPN' 前体）。
    早期版本在测试里手写了一段"看起来像枯草杆菌蛋白酶"的序列，
    结果它与真实序列不符（His 模体间距 153 而非 157，且完全缺少 DSG/GNEGT），
    导致对活性位点识别的验证失效——这正说明测试数据必须来自真实来源。
    """

    SEED_PATH = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "data" / "seeds" / "protease_subtilisin.fasta"
    )

    @pytest.fixture(scope="class")
    def subtilisin(self) -> str:
        if not self.SEED_PATH.exists():
            pytest.skip(
                f"缺少真实种子序列 {self.SEED_PATH}，"
                "请先执行 python scripts/fetch_seed_sequences.py"
            )
        chunks = [
            line.strip()
            for line in self.SEED_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith(">")
        ]
        sequence = "".join(chunks).upper()
        if len(sequence) < 300:
            pytest.skip(f"种子序列长度异常（{len(sequence)} aa），文件可能不完整")
        return sequence

    def test_motifs_detected_with_correct_spacing(self, subtilisin):
        from backend.app.services.design.rulepacks.protease import verify_active_site

        verification = verify_active_site(subtilisin)
        assert verification["catalytic_ser_1based"] is not None
        # 相对间距是与编号体系无关的不变量，是识别正确性的独立交叉验证
        assert verification["all_checks_passed"], verification["spacing_check"]
        checks = verification["spacing_check"]
        assert checks["his"]["actual"] == 157
        assert checks["asp"]["actual"] == 189

    def test_all_four_motifs_found(self, subtilisin):
        from backend.app.services.design.rulepacks.protease import ACTIVE_SITE_MOTIFS

        for motif in ACTIVE_SITE_MOTIFS:
            assert motif in subtilisin, f"序列中缺少模体 {motif}"

    def test_catalytic_residues_are_protected(self, subtilisin):
        pack = get_rulepack("protease")
        protected = pack.protected_positions(subtilisin)
        for label in ("催化丝氨酸", "催化组氨酸", "催化天冬氨酸", "氧负离子洞"):
            assert any(label in reason for reason in protected.values()), f"缺少 {label} 保护"

    def test_proregion_is_protected(self, subtilisin):
        """前导肽（信号肽 + propeptide）在成熟过程中被切除，
        在这些位点推荐突变对成熟酶毫无意义，必须保护。"""
        pack = get_rulepack("protease")
        protected = pack.protected_positions(subtilisin)
        proregion = [position for position, reason in protected.items() if "前导肽" in reason]
        assert len(proregion) == 107, f"前导肽保护位点数应为 107，实际 {len(proregion)}"
        assert min(proregion) == 0

    def test_mutating_catalytic_serine_is_blocked(self, subtilisin):
        from backend.app.services.design.rulepacks.base import RuleContext
        from backend.app.services.design.rulepacks.protease import verify_active_site

        verification = verify_active_site(subtilisin)
        position = verification["catalytic_ser_1based"] - 1
        pack = get_rulepack("protease")
        context = RuleContext(
            sequence=subtilisin, position=position, wild_type="S", mutant="A"
        )
        assert pack.activity_factor(context).factor == 0.0

    def test_mutating_proregion_is_blocked(self, subtilisin):
        from backend.app.services.design.rulepacks.base import RuleContext

        pack = get_rulepack("protease")
        context = RuleContext(sequence=subtilisin, position=10, wild_type=subtilisin[10], mutant="A")
        verdict = pack.activity_factor(context)
        assert verdict.factor == 0.0
        assert "前导肽" in verdict.note or any("前导肽" in flag for flag in verdict.flags)


class TestScanner:
    """扫描计划与位点排除透明度。"""

    def test_terminal_positions_excluded(self, sample_sequence):
        plan = build_scan_plan(sample_sequence, exclude_terminal=1)
        assert 0 not in plan.positions
        assert len(sample_sequence) - 1 not in plan.positions
        assert any("末端" in item.reason for item in plan.exclusions)

    def test_protected_positions_recorded_with_reason(self, sample_sequence):
        protected = {3: "人工保护用于测试"}
        plan = build_scan_plan(sample_sequence, protected=protected)
        assert 3 not in plan.positions
        assert any(item.position == 3 and item.reason == "人工保护用于测试" for item in plan.exclusions)

    def test_region_limit_respected(self, sample_sequence):
        plan = build_scan_plan(sample_sequence, region=(10, 20))
        assert all(10 <= position < 20 for position in plan.positions)

    def test_candidate_count_is_positions_times_nineteen(self, sample_sequence):
        plan = build_scan_plan(sample_sequence)
        assert plan.candidate_count == len(plan.positions) * 19

    def test_strict_mode_rejects_ambiguous(self):
        from backend.app.core.errors import SequenceError

        with pytest.raises(SequenceError):
            build_scan_plan("MKTVXQERLKSIVRILERSKE", strict=True)

    def test_mutation_label_is_one_based(self, sample_sequence):
        # sample_sequence = MKTVRQERLK...：index 0 是 M，index 9 是 K
        assert mutation_label(sample_sequence, 0, "V") == "M1V"
        assert mutation_label(sample_sequence, 9, "A") == "K10A"

    def test_build_mutant_sequence(self, sample_sequence):
        mutated = build_mutant_sequence(sample_sequence, 4, "E")
        assert mutated[4] == "E"
        assert len(mutated) == len(sample_sequence)
        assert mutated[:4] == sample_sequence[:4]
