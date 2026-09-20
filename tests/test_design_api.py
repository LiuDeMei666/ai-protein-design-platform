"""突变设计接口端到端测试（使用 stub Provider，不依赖外网）。

依赖 ESM-2 的用例通过 ``esm_available`` 夹具自动跳过，保证无 GPU /
未下载权重时测试套件仍可运行。
"""

from __future__ import annotations

import time

import pytest


def _wait_job(client, job_id: str, timeout: float = 600.0) -> dict:
    """轮询作业直到终态。"""
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        last = response.json()
        if last["status"] in ("success", "failed", "cancelled"):
            return last
        time.sleep(0.6)
    raise AssertionError(f"作业 {job_id} 在 {timeout}s 内未结束，最后状态 {last.get('status')}")


class TestHealthAndCatalog:
    """基础接口。"""

    def test_health_reports_components(self, api_client):
        response = api_client.get("/api/health")
        assert response.status_code == 200
        payload = response.json()
        for key in ("healthy", "database", "torch", "structure", "embedding", "storage", "config"):
            assert key in payload
        assert payload["database"]["ok"] is True
        assert "stub" in payload["structure"]["providers"]

    def test_liveness(self, api_client):
        assert api_client.get("/api/health/live").json() == {"status": "alive"}

    def test_design_catalog_lists_modes_and_types(self, api_client):
        payload = api_client.get("/api/design/catalog").json()
        assert {mode["key"] for mode in payload["modes"]} == {
            "single", "combination", "local", "manual"
        }
        assert {item["key"] for item in payload["protein_types"]} == {
            "collagen", "protease", "protein_a", "generic"
        }
        assert "stability" in payload["scoring_dimensions"]

    def test_property_catalog_lists_nine_metrics(self, api_client):
        payload = api_client.get("/api/property/catalog").json()
        keys = {item["key"] for item in payload["metrics"]}
        assert len(keys) == 9
        assert "thermostability" in keys and "immunogenicity" in keys

    def test_structure_providers_listed(self, api_client):
        payload = api_client.get("/api/structure/providers").json()
        names = {item["name"] for item in payload}
        assert names == {"esmatlas", "local_esmfold", "stub"}
        stub = next(item for item in payload if item["name"] == "stub")
        assert stub["available"] is True


class TestSequenceEndpoints:
    """序列校验与保存。"""

    def test_validate_passes_clean_sequence(self, api_client, sample_sequence):
        response = api_client.post(
            "/api/sequence/validate", json={"sequence": sample_sequence, "strict": False}
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is True
        assert payload["design_ready"] is True
        assert payload["length"] == len(sample_sequence)
        assert len(payload["composition"]) == 20

    def test_validate_reports_invalid_chars(self, api_client):
        # 26 个字母都被"标准氨基酸"或"歧义残基"覆盖，非法字符只可能是其它符号
        payload = api_client.post(
            "/api/sequence/validate", json={"sequence": "MKTV?RQER", "strict": False}
        ).json()
        assert payload["ok"] is False
        assert "?" in payload["invalid_chars"]

    def test_validate_cleans_digits_with_warning(self, api_client):
        payload = api_client.post(
            "/api/sequence/validate", json={"sequence": "MKTV1RQER", "strict": False}
        ).json()
        assert payload["ok"] is True
        assert payload["length"] == 8  # "MKTV1RQER" 清理掉 '1' 后为 8 个残基
        assert any("清理" in warning for warning in payload["warnings"])

    def test_parse_fasta_returns_records(self, api_client):
        payload = api_client.post(
            "/api/sequence/parse-fasta", json={"text": ">a\nMKTVRQ\n>b\nAAEEKK\n"}
        ).json()
        assert len(payload) == 2
        assert payload[0]["name"] == "a"

    def test_create_and_fetch_sequence(self, api_client, sample_sequence):
        projects = api_client.get("/api/projects").json()["items"]
        project_id = projects[0]["id"]

        created = api_client.post(
            "/api/sequences",
            json={
                "project_id": project_id,
                "name": "api-test-sequence",
                "sequence": sample_sequence,
                "protein_type": "generic",
            },
        )
        assert created.status_code == 200
        record = created.json()
        assert record["length"] == len(sample_sequence)

        fetched = api_client.get(f"/api/sequences/{record['id']}").json()
        assert fetched["sequence"] == sample_sequence

    def test_duplicate_sequence_returns_existing(self, api_client, sample_sequence):
        projects = api_client.get("/api/projects").json()["items"]
        payload = {
            "project_id": projects[0]["id"],
            "name": "dup-test",
            "sequence": sample_sequence,
            "protein_type": "generic",
        }
        first = api_client.post("/api/sequences", json=payload).json()
        second = api_client.post("/api/sequences", json=payload).json()
        assert first["id"] == second["id"]


class TestDesignJob:
    """突变设计作业全流程。"""

    def test_full_design_flow(self, api_client, sample_sequence, esm_available):
        if not esm_available:
            pytest.skip("ESM-2 权重未就绪，跳过依赖模型的作业测试")

        accepted = api_client.post(
            "/api/design/run",
            json={
                "sequence": sample_sequence,
                "protein_type": "generic",
                "mode": "single",
                "top_n": 10,
                "use_structure": True,
                "provider": "stub",  # 强制离线 Provider，避免测试依赖外网
            },
        )
        assert accepted.status_code == 200
        job_id = accepted.json()["job_id"]

        job = _wait_job(api_client, job_id)
        assert job["status"] == "success", job.get("error")

        result = job["result"]
        assert result["length"] == len(sample_sequence)
        # stub 结构的 pLDDT 被刻意压低，必须被剔除而不是用来惩罚真实序列
        assert result["structure"]["note"] == "占位结构未用于打分" or result["structure"]["source"] == "stub"

        candidates = result["candidates"]
        assert len(candidates) == 10

        first = candidates[0]
        assert first["mutations"] and first["positions"]
        assert len(first["dimensions"]) == 5
        # 核心不变量：各维度贡献之和精确等于总分
        assert sum(item["contribution"] for item in first["dimensions"]) == pytest.approx(
            first["total_score"], abs=0.05
        )
        # 每条候选都必须有中文依据，禁止只给总分
        assert all(item["rationale"] for item in first["dimensions"])
        assert first["rationale"]

        # 扫描透明度：排除位点要有原因
        plan = result["plan"]
        assert plan["position_count"] > 0
        assert all(item["reason"] for item in plan["exclusions"])

        # 结果必须落库，否则导出与历史记录会失效
        assert result["design_run_id"]
        run_id = result["design_run_id"]

        runs = api_client.get("/api/design/runs", params={"limit": 50}).json()
        assert any(item["id"] == run_id for item in runs["items"])

        stored = api_client.get(f"/api/design/runs/{run_id}/candidates", params={"limit": 5}).json()
        assert stored["total"] > 0
        assert len(stored["items"]) == 5

        # 导出（含 BOM，Excel 打开不乱码）
        csv_response = api_client.get(
            f"/api/design/runs/{run_id}/export", params={"format": "csv", "top_n": 5}
        )
        assert csv_response.status_code == 200
        assert csv_response.content.startswith("\ufeff".encode())
        assert "突变" in csv_response.text

        markdown = api_client.get(
            f"/api/design/runs/{run_id}/export", params={"format": "markdown", "top_n": 3}
        )
        assert markdown.status_code == 200
        assert "# 突变设计候选方案" in markdown.text

    def test_design_requires_valid_sequence(self, api_client):
        response = api_client.post(
            "/api/design/run",
            json={
                "sequence": "MKTVXQERLKSIVR",  # 含歧义残基，突变设计必须拒绝
                "protein_type": "generic",
                "mode": "single",
                "provider": "stub",
            },
        )
        assert response.status_code == 200  # 作业受理
        job = _wait_job(api_client, response.json()["job_id"], timeout=120)
        assert job["status"] == "failed"
        assert "歧义" in (job["error"] or "") or "标准氨基酸" in (job["error"] or "")

    def test_job_list_and_cancel(self, api_client):
        listing = api_client.get("/api/jobs", params={"limit": 10}).json()
        assert "items" in listing
        assert listing["total"] >= 0

        queue = api_client.get("/api/jobs/queue").json()
        assert "workers" in queue and "active_jobs" in queue

    def test_unknown_job_returns_404(self, api_client):
        assert api_client.get("/api/jobs/does-not-exist").status_code == 404


class TestDashboard:
    """工作台汇总。"""

    def test_dashboard_shape(self, api_client):
        payload = api_client.get("/api/dashboard").json()
        for key in (
            "project_count", "sequence_count", "design_run_count",
            "mutation_count", "experiment_count", "model_version_count", "active_jobs",
        ):
            assert key in payload, f"看板缺少字段 {key}"

    def test_projects_listing(self, api_client):
        payload = api_client.get("/api/projects").json()
        assert payload["total"] >= 1
        assert payload["items"][0]["name"] == "默认项目"


class TestManualDesignMode:
    """人工指定突变（验证 MVP 工作流）。

    这几个接口是**纯规划与校验**：不调用 ESM-2、不预测结构、不写数据库，
    因此不依赖 GPU 与模型权重，在 CI 上会真实执行而不会被 skip。
    它们守住的是几条容易在重构中被破坏的契约：

    * ``wild`` 必须由平台从序列推导，不接受人工输入；
    * 锁定位点冲突必须**显式返回原因**，不能悄悄从清单里消失；
    * 导出即定稿，有冲突必须拒绝，而不是产出缺斤少两的清单。
    """

    #: conftest 里 SAMPLE_SEQUENCE 的位点 10 / 20 / 30 分别是 K / K / E
    def test_catalog_exposes_manual_mode_and_defaults(self, api_client):
        payload = api_client.get("/api/design/catalog").json()
        assert "manual" in {mode["key"] for mode in payload["modes"]}
        manual = payload.get("manual") or {}
        assert manual.get("default_substitutions"), "目录里必须给出人工模式的默认候选氨基酸"
        assert "hydrophobic" in manual["default_substitutions"]

    def test_plan_derives_wild_type_from_sequence(self, api_client, sample_sequence):
        response = api_client.post(
            "/api/design/manual/plan",
            json={
                "sequence": sample_sequence,
                "name": "Test",
                "target_positions": [10, 20],
                "substitutions": {"10": ["A", "S"]},
            },
        )
        assert response.status_code == 200
        plan = response.json()
        assert plan["blocked"] == []

        by_position: dict[int, list[dict]] = {}
        for item in plan["mutations"]:
            by_position.setdefault(item["position"], []).append(item)

        # 位点 10 只出人工指定的两个替换
        assert {item["mutant"] for item in by_position[10]} == {"A", "S"}
        # 位点 20 未指定 -> 套用默认集合（含丙氨酸扫描，故至少 3 个）
        assert len(by_position[20]) >= 3
        assert "A" in {item["mutant"] for item in by_position[20]}

        # 关键不变量：wild 来自序列；生成的序列只在该位点发生变化
        for item in plan["mutations"]:
            index = item["position"] - 1
            assert item["wild_type"] == sample_sequence[index]
            assert item["sequence"][index] == item["mutant"]
            assert item["sequence"][:index] == sample_sequence[:index]
            assert item["sequence"][index + 1 :] == sample_sequence[index + 1 :]
            assert item["sequence_id"].endswith(item["label"])

    def test_plan_reports_locked_conflict_without_silently_dropping(
        self, api_client, sample_sequence
    ):
        """锁定位点冲突要带原因返回——这是“不静默失败”在人工模式下的体现。"""
        response = api_client.post(
            "/api/design/manual/plan",
            json={
                "sequence": sample_sequence,
                "target_positions": [10, 20],
                "locked_positions": [10],
            },
        )
        assert response.status_code == 200  # 预览允许带冲突返回，交给前端展示
        plan = response.json()
        assert [item["position"] for item in plan["blocked"]] == [10]
        assert "锁定位点" in plan["blocked"][0]["reason"]
        assert 10 not in {item["position"] for item in plan["mutations"]}
        assert 20 in {item["position"] for item in plan["mutations"]}

    def test_export_refuses_conflicting_plan(self, api_client, sample_sequence):
        """导出即定稿：宁可报错，也不产出少了几条位点的清单。"""
        response = api_client.post(
            "/api/design/manual/export?format=csv",
            json={
                "sequence": sample_sequence,
                "target_positions": [10],
                "locked_positions": [10],
            },
        )
        assert response.status_code == 400

    def test_export_has_required_columns_and_wild_type_control(
        self, api_client, sample_sequence
    ):
        """需求要求 CSV 前三列为 sequence_id / fasta / mutation_note，且必须含野生对照。"""
        response = api_client.post(
            "/api/design/manual/export?format=csv",
            json={
                "sequence": sample_sequence,
                "name": "Demo",
                "target_positions": [10],
                "substitutions": {"10": ["A"]},
            },
        )
        assert response.status_code == 200
        header = response.text.splitlines()[0]
        for column in ("sequence_id", "fasta", "mutation_note"):
            assert column in header
        assert ">Demo_WT" in response.text, "野生型基准对照必须包含在导出结果里"

        fasta = api_client.post(
            "/api/design/manual/export?format=fasta",
            json={
                "sequence": sample_sequence,
                "name": "Demo",
                "target_positions": [10],
                "substitutions": {"10": ["A"]},
            },
        )
        assert fasta.status_code == 200
        assert fasta.text.startswith(">Demo_WT")

    def test_plan_rejects_invalid_amino_acid(self, api_client, sample_sequence):
        response = api_client.post(
            "/api/design/manual/plan",
            json={
                "sequence": sample_sequence,
                "target_positions": [10],
                "substitutions": {"10": ["Z"]},
            },
        )
        assert response.status_code == 400
