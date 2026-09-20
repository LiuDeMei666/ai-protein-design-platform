"""全局配置。

优先级：环境变量 / .env  >  ``configs/default.yaml``  >  代码内默认值。

设计说明
--------
工作区既有项目普遍采用"配置即代码"（dataclass / 模块常量），本平台需要在
企业内网部署时由运维调整路径与外部依赖开关，因此引入 ``pydantic-settings``：
所有可调项都有代码内默认值，开箱可跑；需要覆盖时再通过 ``.env`` 注入。
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# <root>/backend/app/core/config.py -> parents[3] == <root>
PROJECT_ROOT = Path(__file__).resolve().parents[3]

StructureProviderName = Literal["auto", "esmatlas", "local_esmfold", "stub"]
DeviceName = Literal["auto", "cuda", "cpu"]


class Settings(BaseSettings):
    """平台运行期设置。字段名大写化的 ``DBZ_`` 前缀即为环境变量名。"""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="DBZ_",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- 应用元信息 ----------
    app_name: str = "AI 辅助蛋白设计平台"
    version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8848
    debug: bool = False

    # ---------- 目录（默认全部落在项目内，避免污染工作区） ----------
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    uploads_dir: Path = PROJECT_ROOT / "data" / "uploads"
    seeds_dir: Path = PROJECT_ROOT / "data" / "seeds"
    templates_dir: Path = PROJECT_ROOT / "data" / "templates"
    db_path: Path = PROJECT_ROOT / "data" / "db" / "platform.db"
    models_dir: Path = PROJECT_ROOT / "models"
    logs_dir: Path = PROJECT_ROOT / "logs"
    frontend_dir: Path = PROJECT_ROOT / "frontend"
    config_file: Path = PROJECT_ROOT / "configs" / "default.yaml"

    # ---------- 结构预测 ----------
    # auto = 按 esmatlas -> local_esmfold -> stub 顺序探测，首个可用者生效
    structure_provider: StructureProviderName = "auto"
    esmatlas_url: str = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    esmatlas_timeout: float = 120.0
    esmatlas_retries: int = 3
    # ESM Atlas 单次折叠的实测长度上限约 400 aa，超长序列由 domain_splitter 切分
    esmatlas_max_length: int = 400
    esmatlas_concurrency: int = 2
    local_esmfold_weights: str = "esmfold_3B_v1"

    # ---------- 蛋白语言模型（ESM-2） ----------
    esm_model: str = "facebook/esm2_t33_650M_UR50D"
    esm_fallback_model: str = "facebook/esm2_t12_35M_UR50D"
    device: DeviceName = "auto"
    use_fp16: bool = True
    max_embedding_length: int = 1022  # 模型位置上限 1024，预留 BOS/EOS
    # 实测批大小 2 时 65 位点掩码扫描需 1.27s；提高到 8 可近似线性加速，
    # V100 32GB 在 1022 长度窗口下显存占用仍有余量
    embedding_batch_size: int = 8
    # 实测 huggingface.co 直连失败（SSL），必须走镜像
    hf_endpoint: str = Field(default="https://hf-mirror.com", alias="HF_ENDPOINT")

    # ---------- 序列约束 ----------
    max_sequence_length: int = 5000
    max_upload_mb: float = 20.0

    # ---------- 异步作业 ----------
    job_workers: int = 2
    job_timeout_seconds: int = 3600
    # 全局串行化的重计算开关：为 True 时嵌入/结构预测互斥，避免 V100 显存争抢
    serialize_heavy_jobs: bool = True

    # ---------- 鉴权 ----------
    api_key: str | None = None

    @field_validator(
        "data_dir",
        "cache_dir",
        "uploads_dir",
        "seeds_dir",
        "templates_dir",
        "models_dir",
        "logs_dir",
        mode="after",
    )
    @classmethod
    def _ensure_dir(cls, value: Path) -> Path:
        value.mkdir(parents=True, exist_ok=True)
        return value

    @field_validator("db_path", mode="after")
    @classmethod
    def _ensure_db_parent(cls, value: Path) -> Path:
        value.parent.mkdir(parents=True, exist_ok=True)
        return value

    @property
    def require_api_key(self) -> bool:
        return bool(self.api_key)

    def ensure_runtime_dirs(self) -> None:
        """补齐运行期目录（子目录按需创建，启动自检时调用）。"""
        for sub in ("structures", "embeddings", "pdb_models"):
            (self.cache_dir / sub).mkdir(parents=True, exist_ok=True)

    @property
    def platform(self) -> dict[str, Any]:
        """``configs/default.yaml`` 的内容，见 :func:`load_platform_config`。"""
        return load_platform_config(self.config_file)


@functools.lru_cache(maxsize=8)
def _read_yaml(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_platform_config(path: Path | str | None = None) -> dict[str, Any]:
    """读取平台参数（评分权重、规则包配置、宿主表达体系等）。"""
    target = Path(path) if path is not None else (PROJECT_ROOT / "configs" / "default.yaml")
    return _read_yaml(str(target))


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """全局配置单例。"""
    return Settings()


def reset_settings_cache() -> None:
    """测试与热重载用：清空配置缓存。"""
    get_settings.cache_clear()
    _read_yaml.cache_clear()
