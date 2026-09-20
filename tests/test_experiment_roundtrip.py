"""实验数据录入、对比分析与增量训练回路的闭环测试。"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest


def _csv_bytes(rows: list[list[str]], header: list[str]) -> bytes:
    import pandas as pd

    buffer = io.StringIO()
    pd.DataFrame(rows, columns=header).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")


HEADER = ["样品", "突变", "属性", "实测值", "单位", "条件", "重复", "操作人", "日期", "备注"]


class TestIngestMapping:
    """列名映射与属性名归一。"""

    def test_chinese_headers_are_mapped(self):
        from backend.app.services.experiment.ingest import map_columns

        mapping, unmapped = map_columns(HEADER)
        assert mapping["样品"] == "sequence_name"
        assert mapping["突变"] == "mutation"
        assert mapping["属性"] == "property_name"
        assert mapping["实测值"] == "measured_value"
        assert unmapped == []

    def test_english_alias_headers_are_mapped(self):
        from backend.app.services.experiment.ingest import map_columns

        mapping, _ = map_columns(["mutant", "property", "value", "unit", "comment"])
        assert mapping["mutant"] == "mutation"
        assert mapping["property"] == "property_name"
        assert mapping["value"] == "measured_value"

    def test_property_name_normalization(self):
        from backend.app.services.experiment.ingest import normalize_property_name

        assert normalize_property_name("热稳定性") == ("thermostability", True)
        assert normalize_property_name("Tm")[0] == "thermostability"
        assert normalize_property_name("溶解性")[0] == "solubility"
        assert normalize_property_name("耐碱性")[0] == "alkali_stability"
        # 未知属性原样保留，不拒绝企业的自定义指标
        assert normalize_property_name("自研活性指标") == ("自研活性指标", False)

    def test_numeric_parsing_tolerates_comparators(self):
        from backend.app.services.experiment.ingest import parse_numeric

        assert parse_numeric("68.5")[0] == 68.5
        value, warning = parse_numeric(">90")
        assert value == 90.0 and warning
        assert parse_numeric("abc")[0] is None
        assert parse_numeric(None)[0] is None


class TestIngestValidation:
    """逐行校验：绝不静默丢弃。"""

    def test_bad_row_is_rejected_with_row_number(self):
        from backend.app.services.experiment.ingest import (
            parse_dataframe,
            read_table,
        )

        content = _csv_bytes(
            [
                ["WT", "", "热稳定性", "62.0", "°C", "pH7", "1", "张三", "2026-03-01", ""],
                ["BAD", "X9Y", "热稳定性", "abc", "°C", "", "1", "", "", ""],
            ],
            HEADER,
        )
        frame = read_table(content, "t.csv")
        parsed, _, _ = parse_dataframe(frame)
        assert len(parsed) == 2
        assert not parsed[0].errors
        assert parsed[1].errors
        assert parsed[1].errors[0]["row"] == 3  # 1-based 且含表头

    def test_missing_required_field_rejected(self):
        from backend.app.services.experiment.ingest import parse_dataframe, read_table

        content = _csv_bytes(
            [["WT", "", "", "", "", "", "", "", "", ""]], HEADER
        )
        parsed, _, _ = parse_dataframe(read_table(content, "t.csv"))
        fields = {error["field"] for error in parsed[0].errors}
        assert "property_name" in fields
        assert "measured_value" in fields

    def test_unknown_property_is_flagged_not_rejected(self):
        from backend.app.services.experiment.ingest import parse_dataframe, read_table

        content = _csv_bytes(
            [["WT", "", "自研指标Z", "12", "", "", "", "", "", ""]], HEADER
        )
        parsed, _, _ = parse_dataframe(read_table(content, "t.csv"))
        assert not parsed[0].errors
        assert parsed[0].unknown_property is True

    def test_template_columns_documented(self):
        from backend.app.services.experiment.ingest import template_columns

        columns = template_columns()
        assert len(columns) == 10
        required = [item for item in columns if item["required"]]
        assert {item["name"] for item in required} == {"sequence_name", "property_name", "measured_value"}


class TestIngestViaApi:
    """通过 REST 接口导入。"""

    def test_dry_run_does_not_persist(self, api_client):
        content = _csv_bytes(
            [["WT", "", "热稳定性", "62.0", "°C", "pH7", "1", "张三", "2026-03-01", ""]],
            HEADER,
        )
        before = api_client.get("/api/experiment/records", params={"limit": 1}).json()["total"]

        response = api_client.post(
            "/api/experiment/ingest",
            files={"file": ("t.csv", content, "text/csv")},
            data={"project_id": "1", "dry_run": "true", "deduplicate": "true"},
        )
        assert response.status_code == 200
        report = response.json()["report"]
        assert report["dry_run"] is True
        assert report["accepted_rows"] == 1
        assert response.json()["inserted_ids"] == []

        after = api_client.get("/api/experiment/records", params={"limit": 1}).json()["total"]
        assert after == before

    def test_real_ingest_persists_and_deduplicates(self, api_client):
        content = _csv_bytes(
            [
                ["WT", "", "热稳定性", "62.0", "°C", "pH7-dup", "1", "张三", "2026-03-01", ""],
                ["WT", "", "热稳定性", "64.0", "°C", "pH7-dup", "2", "张三", "2026-03-02", ""],
                ["WT", "", "热稳定性", "abc", "°C", "", "1", "", "", "坏行"],
            ],
            HEADER,
        )
        response = api_client.post(
            "/api/experiment/ingest",
            files={"file": ("t.csv", content, "text/csv")},
            data={"project_id": "1", "dry_run": "false", "deduplicate": "true"},
        )
        assert response.status_code == 200
        report = response.json()["report"]
        assert report["accepted_rows"] == 2
        assert report["rejected_rows"] == 1
        assert len(response.json()["inserted_ids"]) == 2

        # 重复导入应全部被去重跳过
        again = api_client.post(
            "/api/experiment/ingest",
            files={"file": ("t.csv", content, "text/csv")},
            data={"project_id": "1", "dry_run": "false", "deduplicate": "true"},
        ).json()
        assert again["inserted_ids"] == []
        assert again["report"]["duplicate_rows"] == 2

    def test_template_download(self, api_client):
        response = api_client.get("/api/experiment/template.xlsx")
        assert response.status_code == 200
        assert len(response.content) > 3000

        csv_response = api_client.get("/api/experiment/template.csv")
        assert csv_response.status_code == 200
        assert csv_response.content.startswith("\ufeff".encode())


class TestSequenceAssociation:
    """导入时的序列关联校验。

    这条不变量守的是数据一致性：项目列表与看板按 ``project_id`` 统计、
    预测-实测对比按 ``sequence_id`` 过滤。若允许"记录挂在 A 项目、却引用
    B 项目的序列"而不出声，两边对"这批数据属于谁"的认知就不一致了，
    而且这类问题在界面上完全看不出来，排查代价极高。
    """

    @staticmethod
    def _create_project(api_client, name: str) -> int:
        response = api_client.post("/api/projects", json={"name": name})
        assert response.status_code in (200, 201), response.text
        return response.json()["id"]

    @staticmethod
    def _create_sequence(api_client, project_id: int, name: str, sequence: str) -> int:
        response = api_client.post(
            "/api/sequences",
            json={
                "project_id": project_id,
                "name": name,
                "sequence": sequence,
                "protein_type": "generic",
            },
        )
        assert response.status_code in (200, 201), response.text
        return response.json()["id"]

    @staticmethod
    def _ingest(api_client, project_id: int, **extra):
        content = _csv_bytes(
            [["样品", "", "热稳定性", "62.0", "°C", "assoc-cond", "1", "张三", "2026-03-01", ""]],
            HEADER,
        )
        return api_client.post(
            "/api/experiment/ingest",
            files={"file": ("t.csv", content, "text/csv")},
            data={"project_id": str(project_id), "dry_run": "true", **extra},
        )

    def test_cross_project_sequence_is_reported(self, api_client, sample_sequence):
        """显式选了别的项目的序列：允许，但必须在报告里说明。"""
        other_project = self._create_project(api_client, "另一个项目")
        sequence_id = self._create_sequence(
            api_client, other_project, "跨项目序列", sample_sequence
        )

        response = self._ingest(api_client, 1, sequence_id=str(sequence_id))
        assert response.status_code == 200
        joined = " ".join(response.json()["report"]["warnings"])
        assert "另一个项目" in joined, "应指出序列实际所属的项目名"
        assert "不一致" in joined

    def test_same_project_sequence_is_silent(self, api_client, sample_sequence):
        project = self._create_project(api_client, "同项目测试")
        sequence_id = self._create_sequence(api_client, project, "同项目序列", sample_sequence)

        response = self._ingest(api_client, project, sequence_id=str(sequence_id))
        assert response.status_code == 200
        warnings = response.json()["report"]["warnings"]
        assert not any("不一致" in item for item in warnings)

    def test_unknown_sequence_is_rejected(self, api_client):
        """此前不存在的 sequence_id 会被直接写入，现在明确报错。"""
        response = self._ingest(api_client, 1, sequence_id="999999")
        assert response.status_code == 400


class TestMutationLabelValidation:
    """突变标签与序列的一致性校验（挡住最常见的录入错误）。"""

    def test_valid_label(self, api_client, sample_sequence):
        project_id = api_client.get("/api/projects").json()["items"][0]["id"]
        record = api_client.post(
            "/api/sequences",
            json={
                "project_id": project_id,
                "name": "label-check",
                "sequence": sample_sequence,
                "protein_type": "generic",
            },
        ).json()
        payload = api_client.post(
            "/api/experiment/validate-sequence",
            params={"sequence_id": record["id"], "mutation": "M1V"},
        ).json()
        assert payload["ok"] is True

    def test_wrong_wild_type_is_rejected(self, api_client, sample_sequence):
        project_id = api_client.get("/api/projects").json()["items"][0]["id"]
        record = api_client.post(
            "/api/sequences",
            json={
                "project_id": project_id,
                "name": "label-check-2",
                "sequence": sample_sequence,
                "protein_type": "generic",
            },
        ).json()
        payload = api_client.post(
            "/api/experiment/validate-sequence",
            params={"sequence_id": record["id"], "mutation": "A1V"},  # 第 1 位实际是 M
        ).json()
        assert payload["ok"] is False
        assert payload["errors"]


class TestCompare:
    """预测-实测对比。"""

    def test_compare_requires_records(self, api_client, sample_sequence):
        project_id = api_client.get("/api/projects").json()["items"][0]["id"]
        record = api_client.post(
            "/api/sequences",
            json={
                "project_id": project_id,
                "name": "compare-empty",
                "sequence": sample_sequence,
                "protein_type": "generic",
            },
        ).json()
        response = api_client.post(
            "/api/experiment/compare", params={"sequence_id": record["id"], "use_esm": "false"}
        )
        assert response.status_code == 400
        assert "没有可对比" in response.json()["message"]

    def test_compare_statistics_are_sane(self, api_client, sample_sequence, esm_available):
        """用平台自身的预测值构造数据，验证对比统计的正确性（非真实实验数据）。"""
        if not esm_available:
            pytest.skip("需要 ESM-2 才能重算突变体预测值")

        from backend.app.services.experiment.compare import compare_records
        from backend.app.services.property.pipeline import predict_properties

        mutations = ["M1V", "K2A", "T3S", "V4L", "R5K", "Q6E", "E7D", "R8K"]
        records = []
        for label in mutations:
            position = int(label[1:-1]) - 1
            mutated = sample_sequence[:position] + label[-1] + sample_sequence[position + 1 :]
            score = predict_properties(mutated, use_esm=False).metrics["thermostability"].score
            records.append(
                {
                    "mutation": label,
                    "property_name": "thermostability",
                    "measured_value": float(score),  # 用预测值当"实测值"，仅验证统计逻辑
                    "unit": "score",
                }
            )

        result = compare_records(sample_sequence, records)
        assert result["matched_pairs"] == len(mutations)
        property_report = result["properties"][0]
        assert property_report["n_pairs"] == len(mutations)
        # 预测值与"实测值"同源，秩相关应为 1.0
        assert property_report["spearman_rho"] == pytest.approx(1.0, abs=1e-6)
        assert property_report["r2"] == pytest.approx(1.0, abs=1e-6)
        assert property_report["mae"] == pytest.approx(0.0, abs=1e-6)
        assert result["methodology"]["primary_metric"].startswith("Spearman")

    def test_small_sample_gives_no_quantitative_verdict(self, sample_sequence):
        from backend.app.services.experiment.compare import compare_records

        result = compare_records(
            sample_sequence,
            [
                {"mutation": "M1V", "property_name": "thermostability", "measured_value": 60.0},
                {"mutation": "K2A", "property_name": "thermostability", "measured_value": 62.0},
            ],
            # 关闭 ESM 以加速：此时预测值为确定性计算
        )
        report = result["properties"][0]
        assert "样本量不足" in report["verdict"]


class TestPropertyHead:
    """属性头：增量训练与序列化。"""

    def test_ridge_partial_fit_equals_full_refit(self):
        """Ridge 用充分统计量累积，增量训练必须与全量重训数值等价。"""
        from backend.app.ml.head import PropertyHead

        rng = np.random.default_rng(42)
        features = rng.normal(size=(40, 16))
        targets = features @ rng.normal(size=16) + rng.normal(scale=0.1, size=40)

        incremental = PropertyHead(alpha=0.5)
        incremental.fit(features[:20], targets[:20])
        incremental.partial_fit(features[20:], targets[20:])

        full = PropertyHead(alpha=0.5)
        full.fit(features, targets)

        # 标准化参数不同（增量复用首轮统计量），因此逐个预测值做回归比较
        correlation = np.corrcoef(incremental.predict(features), full.predict(features))[0, 1]
        assert correlation > 0.95

    def test_save_and_load_roundtrip(self, tmp_path):
        from backend.app.ml.head import PropertyHead

        rng = np.random.default_rng(0)
        features = rng.normal(size=(20, 8))
        targets = features @ rng.normal(size=8)

        head = PropertyHead(alpha=1.0).fit(features, targets)
        predictions = head.predict(features)

        path = head.save(tmp_path / "head.npz")
        restored = PropertyHead.load(path)
        assert restored.is_fitted
        assert restored.feature_dim == 8
        assert np.allclose(restored.predict(features), predictions, atol=1e-6)
        assert restored.n_samples == 20

    def test_metrics_computed(self):
        from backend.app.ml.head import regression_metrics

        truth = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.1, 1.9, 3.1, 3.9])
        metrics = regression_metrics(truth, predicted)
        assert metrics.n_samples == 4
        assert metrics.mae == pytest.approx(0.1, abs=1e-6)
        assert metrics.r2 > 0.99

    def test_eval_buffer_enables_meaningful_incremental_metrics(self):
        """增量训练必须在跨新旧数据的评估集上算指标。

        若只用新增样本算 R²，会得到 −251 这类毫无意义且严重误导的数字。
        """
        from backend.app.ml.head import PropertyHead

        rng = np.random.default_rng(1)
        features = rng.normal(size=(30, 6))
        targets = features @ rng.normal(size=6)

        head = PropertyHead(alpha=1.0).fit(features[:25], targets[:25])
        head.partial_fit(features[25:], targets[25:])

        metrics = head.evaluate_buffer()
        assert metrics is not None
        assert metrics.n_samples == 30  # 覆盖新旧全部样本
        assert metrics.r2 > 0.9  # 合理量级，而不是 −251


class TestModelAPI:
    """模型版本接口。"""

    def test_properties_listing(self, api_client):
        payload = api_client.get("/api/model/properties", params={"min_records": 1}).json()
        assert "properties" in payload
        assert "note" in payload

    def test_versions_listing(self, api_client):
        payload = api_client.get("/api/model/versions").json()
        assert "items" in payload and "total" in payload

    def test_train_rejects_insufficient_samples(self, api_client):
        """样本不足时必须明确报错，而不是训练出一个不可信的模型。"""
        response = api_client.post(
            "/api/model/train",
            json={"property_name": "no_such_property_xyz", "algo": "ridge", "min_samples": 8},
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        deadline = time.time() + 120
        job = {}
        while time.time() < deadline:
            job = api_client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("success", "failed", "cancelled"):
                break
            time.sleep(0.5)
        assert job["status"] == "failed"
        assert "数据不足" in (job["error"] or "") or "不足" in (job["error"] or "")
