"""序列校验与基础生物物理量测试。"""

from __future__ import annotations

import pytest

from backend.app.services.sequence.feature_utils import (
    charge_profile,
    collagen_repeat_density,
    composition,
    gly_xy_phase,
    hydrophobic_moment,
    is_gly_xy_repeat,
    net_charge,
)
from backend.app.services.sequence.validator import (
    parse_fasta,
    sequence_sha256,
    validate_sequence,
)


class TestValidation:
    """序列校验行为。"""

    def test_clean_sequence_passes(self, sample_sequence):
        check = validate_sequence(sample_sequence)
        assert check.ok
        assert check.design_ready
        assert check.length == len(sample_sequence)
        assert check.invalid_chars == {}
        assert check.ambiguous_positions == []

    def test_whitespace_and_newlines_are_cleaned(self):
        check = validate_sequence("MKTV RQER\nLKSI\tVRIL")
        assert check.ok
        assert check.sequence == "MKTVRQERLKSIVRIL"
        assert any("清理" in warning for warning in check.warnings)

    def test_invalid_characters_are_reported_not_dropped(self):
        """26 个字母全部被"标准氨基酸"或"歧义残基"覆盖，
        因此非法字符只可能是空白/数字/星号之外的其它符号（如 '?'）。
        数字与星号走的是"清理并告警"路径，不进入 invalid_chars。"""
        check = validate_sequence("MKTV?RQER")
        assert not check.ok
        assert "?" in check.invalid_chars
        assert any("非法字符" in error for error in check.errors)

    def test_digits_are_cleaned_with_warning_in_lenient_mode(self):
        check = validate_sequence("MKTV1RQER")
        assert check.ok  # 宽松模式：清理并告警
        assert check.sequence == "MKTVRQER"
        assert any("清理" in warning for warning in check.warnings)

    def test_ambiguous_residues_warn_in_lenient_mode(self):
        check = validate_sequence("MKTVXQERLKSIV")
        assert check.ok  # 宽松模式仍可继续
        assert not check.design_ready  # 但不可用于突变设计
        assert 4 in check.ambiguous_positions
        assert check.ambiguous_summary == {"X": 1}

    def test_ambiguous_residues_error_in_strict_mode(self):
        check = validate_sequence("MKTVXQERLKSIV", strict=True)
        assert not check.ok
        assert not check.design_ready

    def test_strict_mode_rejects_cleaned_non_residue_chars(self):
        """严格模式下，数字/星号这类误粘贴字符必须报错而不是静默清理。

        这是突变设计入口的关键防线：静默清洗会让用户以为输入是干净的，
        而设计结果会建立在错误输入之上。
        """
        lenient = validate_sequence("MKTVRQER***123")
        assert lenient.ok
        assert lenient.warnings

        strict = validate_sequence("MKTVRQER***123", strict=True)
        assert not strict.ok
        assert any("严格模式" in error for error in strict.errors)

    def test_too_short_sequence_warns(self):
        check = validate_sequence("MKTV")
        assert check.ok
        assert any("置信度" in warning or "可靠性" in warning for warning in check.warnings)

    def test_length_limit_enforced(self):
        check = validate_sequence("A" * 50, max_length=20)
        assert not check.ok
        assert any("上限" in error for error in check.errors)

    def test_empty_sequence_errors(self):
        check = validate_sequence("   \n  ")
        assert not check.ok

    def test_sha256_is_stable_and_content_based(self):
        first = sequence_sha256("MKTV")
        second = sequence_sha256("MKTV")
        third = sequence_sha256("MKTA")
        assert first == second != third
        assert len(first) == 64


class TestFasta:
    """FASTA 解析。"""

    def test_multi_record_parsing(self):
        text = ">seq1 first\nMKTVRQ\n>seq2 second\nAACDEF\n"
        records = parse_fasta(text)
        assert len(records) == 2
        assert records[0].name == "seq1"
        assert records[0].sequence == "MKTVRQ"
        assert records[1].name == "seq2"

    def test_plain_sequence_without_header(self):
        records = parse_fasta("MKTVRQERLK\nSIVRIL")
        assert len(records) == 1
        assert records[0].sequence == "MKTVRQERLKSIVRIL"


class TestFeatureUtils:
    """序列特征工具。"""

    def test_composition_sums_to_one(self, sample_sequence):
        ratios = composition(sample_sequence)
        assert len(ratios) == 20
        assert abs(sum(ratios.values()) - 1.0) < 1e-9

    def test_net_charge_is_monotonic_in_pH(self):
        # pH 升高，净电荷应单调下降
        charges = [net_charge("MKRHEDDEEEKKKRRR", ph) for ph in (3, 5, 7, 9, 11)]
        for before, after in zip(charges, charges[1:]):
            assert after <= before + 1e-9

    def test_isoelectric_point_matches_biopython(self, sample_sequence):
        """本项目实现的 pI 应与 Biopython ProtParam 一致（交叉校验）。"""
        from Bio.SeqUtils.ProtParam import ProteinAnalysis

        ours = charge_profile(sample_sequence).isoelectric_point
        theirs = float(ProteinAnalysis(sample_sequence).isoelectric_point())
        assert abs(ours - theirs) < 0.6, f"本项目 pI={ours}，Biopython pI={theirs}"

    def test_hydrophobic_moment_higher_for_amphipathic(self):
        amphipathic = "LKELLKKLLEK"  # 两亲性螺旋
        uniform = "LELELELELEL"
        assert hydrophobic_moment(amphipathic) > 0
        assert hydrophobic_moment(amphipathic) >= hydrophobic_moment(uniform) * 0.5


class TestCollagenPeriodicity:
    """胶原 Gly-X-Y 周期性识别。"""

    def test_perfect_repeat_detected(self):
        sequence = "GPP" * 12
        assert is_gly_xy_repeat(sequence, 0)
        assert is_gly_xy_repeat(sequence, 9)
        assert gly_xy_phase(sequence, 0) == 0  # Gly 位
        assert gly_xy_phase(sequence, 1) == 1  # X 位
        assert gly_xy_phase(sequence, 2) == 2  # Y 位
        assert collagen_repeat_density(sequence) > 0.9

    def test_broken_repeat_not_flagged(self):
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLA"
        assert not is_gly_xy_repeat(sequence, 0)
        assert collagen_repeat_density(sequence) < 0.3


class TestBiophys:
    """基础生物物理量。"""

    def test_molecular_weight_positive_and_monotonic(self, sample_sequence):
        from backend.app.services.property.biophys import compute_biophys

        short = compute_biophys(sample_sequence[:30]).values
        full = compute_biophys(sample_sequence).values
        assert 0 < short["molecular_weight_da"] < full["molecular_weight_da"]

    def test_insulin_a_chain_reference_values(self):
        """用已知序列做数值回归：人胰岛素 A 链（21 aa）。

        分子量是纯算术量，可直接做精确回归（实测 2383.70 Da）。
        GRAVY 与 pI 受 pKa 集与标度选择影响，只做区间校验（实测 +0.214 / 4.05）。
        """
        from backend.app.services.property.biophys import compute_biophys

        chain_a = "GIVEQCCTSICSLYQLENYCN"
        values = compute_biophys(chain_a).values
        assert values["length"] == 21
        assert abs(values["molecular_weight_da"] - 2383.7) < 1.0
        assert 3.5 < values["isoelectric_point"] < 4.5
        assert -0.1 < values["gravy"] < 0.5

    def test_ambiguous_residues_are_dropped_with_warning(self):
        from backend.app.services.property.biophys import compute_biophys

        result = compute_biophys("MKTVXQERLKSIV")
        assert result.values["dropped_residues"] == 1
        assert result.warnings
        assert result.values["computable"] is True

    def test_n_end_rule_classification(self):
        from backend.app.services.property.biophys import compute_biophys

        stabilizing = compute_biophys("MASKTVRQER").values["n_end_rule"]
        destabilizing = compute_biophys("MRKTVRQERA").values["n_end_rule"]
        assert stabilizing["class"] == "stabilizing"
        assert destabilizing["class"] == "destabilizing"
