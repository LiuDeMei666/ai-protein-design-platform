"""结构预测 Provider、域切分与 PDB 解析测试（全部离线，不依赖外网）。"""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.services.structure import pdb_utils
from backend.app.services.structure.base import StructureResult
from backend.app.services.structure.domain_splitter import (
    merge_fragment_pdbs,
    plan_split,
)
from backend.app.services.structure.registry import (
    get_provider,
    predict_structure,
    provider_status,
    resolve_chain,
)


class TestPlddtScale:
    """pLDDT 量纲归一化（实测差异：ESM Atlas 返回 0-1 分数）。"""

    def test_fraction_scale_detected_and_scaled(self):
        assert pdb_utils.plddt_scale_factor([0.54, 0.93, 0.96]) == 100.0

    def test_hundred_scale_kept(self):
        assert pdb_utils.plddt_scale_factor([37.5, 93.9, 45.0]) == 1.0

    def test_empty_returns_neutral(self):
        assert pdb_utils.plddt_scale_factor([]) == 1.0

    def test_scaling_applied_to_structure(self):
        """ESM Atlas 的 B-factor 0.9 应被归一化为 90 分，而不是 0.9 分。"""
        from backend.app.services.structure.stub import _format_atom_line

        lines = ["HEADER    TEST"]
        for index, residue in enumerate("MKTVR"):
            lines.append(_format_atom_line(index + 1, "CA", "ALA", index + 1, 0.0, 0.0, index * 3.8, 0.9))
        lines.append("END")
        data = pdb_utils.parse_pdb("\n".join(lines))
        assert data.length == 5
        assert all(abs(value - 90.0) < 1e-6 for value in pdb_utils.extract_plddt(data))


class TestStubProvider:
    """离线占位 Provider 的输出必须可解析且"一眼可辨"。"""

    def test_stub_is_always_available(self):
        assert get_provider("stub").available() is True

    def test_stub_produces_parseable_structure(self, stub_structure, sample_sequence):
        result = stub_structure
        assert result.source == "stub"
        assert result.length == len(sample_sequence)
        assert len(result.plddt) == len(sample_sequence)
        assert result.segments == [(0, len(sample_sequence))]
        assert result.truncated is False

        data = pdb_utils.parse_pdb(result.pdb_text)
        assert data.length == len(sample_sequence)
        assert data.sequence == sample_sequence
        # 每个残基都必须有 CA，否则 pLDDT 链路会断裂
        assert all("CA" in residue.atoms for residue in data.residues)

    def test_stub_plddt_is_deliberately_low(self, stub_structure):
        """占位结构的置信度被刻意压低到 30-45，避免被误当成真实预测。"""
        assert 25.0 <= min(stub_structure.plddt)
        assert max(stub_structure.plddt) <= 46.0
        assert 30.0 <= stub_structure.mean_plddt <= 45.0

    def test_stub_marks_degradation(self, stub_structure):
        assert stub_structure.degradation_reason
        assert "占位" in stub_structure.degradation_reason
        assert stub_structure.stats.get("stub") is True


class TestHbondEnergy:
    """氢键能量判据的方向约定（曾是实际踩过的 bug）。"""

    def test_canonical_hbond_geometry_is_negative(self):
        """典型 N-H···O=C 氢键：H 位于 N 与 O 之间时应判为成键。"""
        nitrogen = np.array([0.0, 0.0, 0.0])
        hydrogen = np.array([1.0, 0.0, 0.0])
        oxygen = np.array([2.4, 0.9, 0.0])
        carbon = np.array([3.4, 1.4, 0.0])
        energy = pdb_utils._hbond_energy(nitrogen, hydrogen, carbon, oxygen)
        assert energy < -0.5, f"氢键能量 {energy} 应为明显负值"

    def test_broken_geometry_is_not_a_bond(self):
        """H 背离 O 时不应判为氢键。"""
        nitrogen = np.array([0.0, 0.0, 0.0])
        hydrogen = np.array([-1.0, 0.0, 0.0])  # 反方向
        oxygen = np.array([3.0, 0.0, 0.0])
        carbon = np.array([4.2, 0.0, 0.0])
        assert pdb_utils._hbond_energy(nitrogen, hydrogen, carbon, oxygen) > -0.5


class TestDsspLite:
    """二级结构指派的基本不变量。"""

    def test_length_matches_residue_count(self, stub_structure):
        data = pdb_utils.parse_pdb(stub_structure.pdb_text)
        secondary = pdb_utils.dssp_lite(data)
        assert len(secondary) == data.length
        assert set(secondary) <= set("HGIE-")

    def test_composition_sums_to_one(self):
        composition = pdb_utils.secondary_structure_composition("HHHHEE---GG")
        total = (
            composition["helix"]
            + composition["helix_310"]
            + composition["helix_pi"]
            + composition["sheet"]
            + composition["coil"]
        )
        # 各分量按 4 位小数取整，累加会有 ~1e-4 量级的舍入残差
        assert abs(total - 1.0) < 1e-3

    def test_helix_total_excludes_double_counting(self):
        """helix_total 是 H+G+I 之和，校验时不能再把它加进总和。"""
        composition = pdb_utils.secondary_structure_composition("HHHHGGII--")
        expected = composition["helix"] + composition["helix_310"] + composition["helix_pi"]
        assert composition["helix_total"] == pytest.approx(expected, abs=1e-4)


class TestStructureStats:
    """结构统计的完整性与合理性。"""

    def test_stats_contain_required_fields(self, stub_structure):
        stats = stub_structure.stats
        for key in (
            "plddt_bands",
            "secondary_structure",
            "secondary_structure_composition",
            "radius_of_gyration",
            "mean_relative_sasa",
            "hydrophobic_exposure",
            "plddt",
            "relative_sasa",
            "algorithms",
        ):
            assert key in stats, f"结构统计缺少字段 {key}"

    def test_per_residue_arrays_align_with_length(self, stub_structure, sample_sequence):
        stats = stub_structure.stats
        assert len(stats["plddt"]) == len(sample_sequence)
        assert len(stats["relative_sasa"]) == len(sample_sequence)

    def test_relative_sasa_in_reasonable_range(self, stub_structure):
        values = stub_structure.stats["relative_sasa"]
        assert all(0.0 <= value <= 3.0 for value in values)
        assert np.mean(values) > 0.05  # 不应全为 0

    def test_radius_of_gyration_positive(self, stub_structure):
        assert stub_structure.stats["radius_of_gyration"] > 0


class TestDomainSplitter:
    """长序列域切分。"""

    def test_short_sequence_needs_no_split(self, sample_sequence):
        plan = plan_split(sample_sequence, max_length=400)
        assert plan.count == 1
        assert plan.segments == [(0, len(sample_sequence))]

    def test_long_sequence_is_fully_covered(self):
        sequence = "GPP" * 500  # 1500 aa
        plan = plan_split(sequence, max_length=400)
        assert plan.count >= 4
        assert plan.covers(len(sequence)), "切分必须无缝无重叠地覆盖整条序列"
        assert all(end - start <= 400 for start, end in plan.segments)

    def test_no_segment_exceeds_limit_for_random_sequence(self):
        rng = np.random.default_rng(7)
        alphabet = list("ACDEFGHIKLMNPQRSTVWY")
        sequence = "".join(rng.choice(alphabet, size=1234))
        plan = plan_split(sequence, max_length=300)
        assert plan.covers(len(sequence))
        assert max(end - start for start, end in plan.segments) <= 300

    def test_merge_fragment_pdbs_renumbers_residues(self):
        from backend.app.services.structure.stub import StubPredictor

        predictor = StubPredictor()
        first = predictor.predict("MKTVRQERLK")
        second = predictor.predict("SIVRILERSK")
        merged = merge_fragment_pdbs([first.pdb_text, second.pdb_text], [(0, 10), (10, 20)])
        data = pdb_utils.parse_pdb(merged)
        assert data.length == 20
        # 第二段的残基号必须被偏移到 10+，否则前端热点映射会错位
        assert data.residues[-1].number == 20


class TestRegistry:
    """Provider 注册与降级链。"""

    def test_auto_chain_order(self):
        chain = resolve_chain("auto")
        assert [provider.name for provider in chain] == ["esmatlas", "local_esmfold", "stub"]

    def test_explicit_provider_goes_first(self):
        chain = resolve_chain("stub")
        assert chain[0].name == "stub"
        # 显式指定后仍保留降级后备
        assert len(chain) == 3

    def test_provider_status_shape(self):
        status = provider_status()
        assert set(status.keys()) == {"esmatlas", "local_esmfold", "stub"}
        for name, detail in status.items():
            assert "available" in detail
            assert "max_length" in detail

    def test_explicit_stub_used_without_network(self, sample_sequence):
        result = predict_structure(sample_sequence, provider="stub", use_cache=False)
        assert result.source == "stub"

    def test_esmatlas_has_length_limit(self):
        assert get_provider("esmatlas").max_length == 400
        assert get_provider("stub").max_length is None


class TestStructureResultSummary:
    """结果摘要的字段完整性。"""

    def test_summary_excludes_pdb_text(self, stub_structure):
        summary = stub_structure.to_summary()
        assert "pdb_text" not in summary
        assert "plddt" not in summary
        assert summary["source"] == "stub"

    def test_plddt_bands_sum_to_one(self):
        result = StructureResult(
            pdb_text="",
            plddt=[95.0, 80.0, 60.0, 30.0],
            mean_plddt=66.25,
            source="test",
        )
        bands = result.plddt_bands()
        assert abs(sum(bands.values()) - 1.0) < 1e-6
        assert bands["very_high"] == 0.25
        assert bands["very_low"] == 0.25
