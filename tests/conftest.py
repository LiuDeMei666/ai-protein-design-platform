"""pytest 夹具。

关键设计
--------
1. **隔离运行环境**：在导入任何应用模块之前把数据目录、缓存、数据库都指向
   临时目录，避免测试污染真实的 `data/` 与 `models/`（尤其不能碰企业已录入的数据）。
2. **不依赖外网**：结构预测测试一律使用 `stub` Provider；
   需要 ESM-2 的测试通过 `esm_available` 夹具自动跳过，保证无 GPU 环境也能跑通大部分用例。
3. **确定性**：所有测试使用固定的短序列与固定随机种子。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 必须在导入 backend.* 之前设置环境变量：get_settings() 是 lru_cache 的，
# 一旦被首次调用就会固化目录，之后再改环境变量不会生效。
# ---------------------------------------------------------------------------
_TEMP_ROOT = Path(tempfile.mkdtemp(prefix="dbz-test-"))

os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ["DBZ_DATA_DIR"] = str(_TEMP_ROOT / "data")
os.environ["DBZ_CACHE_DIR"] = str(_TEMP_ROOT / "data" / "cache")
os.environ["DBZ_UPLOADS_DIR"] = str(_TEMP_ROOT / "data" / "uploads")
os.environ["DBZ_SEEDS_DIR"] = str(_TEMP_ROOT / "data" / "seeds")
os.environ["DBZ_TEMPLATES_DIR"] = str(_TEMP_ROOT / "data" / "templates")
os.environ["DBZ_DB_PATH"] = str(_TEMP_ROOT / "data" / "db" / "test.db")
# 注意：models 目录**必须指向项目真实的 models/**，不能指向临时目录。
# 因为 ESM-2 权重（约 2.5 GB）缓存在那里，指向临时目录会让所有依赖模型的用例
# 被静默跳过（esm_available 返回 False），测试覆盖率凭空下降却毫无提示。
# 测试只读取权重；唯一的写入路径（属性头产物）在本套件的用例中不会发生。
_PROJECT_MODELS = Path(__file__).resolve().parents[1] / "models"
_PROJECT_MODELS.mkdir(parents=True, exist_ok=True)
os.environ["DBZ_MODELS_DIR"] = str(_PROJECT_MODELS)
os.environ["DBZ_LOGS_DIR"] = str(_TEMP_ROOT / "logs")
os.environ["DBZ_DEBUG"] = "false"
os.environ["DBZ_JOB_WORKERS"] = "1"
os.environ["DBZ_USE_FP16"] = "true"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


#: 供测试使用的短序列（65 aa 螺旋束，ESM Atlas 实测可折叠）
SAMPLE_SEQUENCE = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
#: 含 Asn-Gly 脱酰胺位点与 Asp-Pro 酸敏感键的测试序列
RISKY_SEQUENCE = "MKTVRQERLKSIVRILERSKEPVSGAQLANGDSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """测试结束后清理临时目录。"""
    shutil.rmtree(_TEMP_ROOT, ignore_errors=True)


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """把请求了 ``esm_available`` 的用例自动标记为 ``slow``。

    背景
    ----
    ``pytest.ini`` 里声明了 ``slow`` 标记（"需要 GPU / ESM-2 权重 / 耗时较长的用例"），
    但全项目**没有任何用例真正打过它**——它一直是个死配置。结果是：
    想排除这批用例只能依赖"机器上没有权重 -> 夹具静默 skip"，
    而 skip 不像 skip，看起来和"通过"没什么两样，覆盖率下降无人察觉。

    在这里按夹具依赖自动打标记，好处是**新增用例时不必记得手工加标记**：
    ``-m "not slow"`` 会永远准确等价于"不依赖模型的那批"，
    不会随代码演进而失准。

    用法：
        pytest -m "not slow"     # CI / 无 GPU 机器：只跑不依赖模型的用例
        pytest -m slow           # 部署机 / 有 GPU：只跑需要权重的用例
        pytest                   # 全跑（缺权重时那批会 skip，并用 -rs 打印原因）
    """
    for item in items:
        if "esm_available" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def sample_sequence() -> str:
    """标准测试序列。"""
    return SAMPLE_SEQUENCE


@pytest.fixture(scope="session")
def risky_sequence() -> str:
    """含已知风险基序的测试序列。"""
    return RISKY_SEQUENCE


@pytest.fixture(scope="session")
def temp_root() -> Path:
    return _TEMP_ROOT


@pytest.fixture(scope="session")
def db_ready():
    """初始化测试数据库（会话级，只建一次）。"""
    from backend.app.db.init_db import init_db

    init_db()
    return True


@pytest.fixture()
def session(db_ready):
    """数据库会话夹具（每个用例独立事务，用例结束回滚）。"""
    from backend.app.db.base import get_session_factory

    db = get_session_factory()()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


@pytest.fixture(scope="session")
def stub_structure(sample_sequence):
    """用 stub Provider 生成的结构结果（会话级缓存，避免重复计算）。"""
    from backend.app.services.structure.registry import predict_structure

    return predict_structure(sample_sequence, provider="stub", use_cache=False)


@pytest.fixture(scope="session")
def esm_available() -> bool:
    """ESM-2 权重是否已就绪（未就绪时跳过依赖模型的用例）。"""
    try:
        from backend.app.services.embedding.esm2 import embedding_status

        status = embedding_status()
        return bool(status.get("main_model_cached") or status.get("fallback_model_cached"))
    except Exception:
        return False


@pytest.fixture()
def api_client(db_ready):
    """FastAPI 测试客户端。"""
    from fastapi.testclient import TestClient

    from backend.app.main import app

    with TestClient(app) as client:
        yield client
