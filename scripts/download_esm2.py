#!/usr/bin/env python
"""下载 ESM-2 权重到项目独立缓存目录。

为什么需要这个脚本
------------------
1. 实测 ``huggingface.co`` 直连失败（curl exit 35，SSL 连接错误），必须走镜像
   ``hf-mirror.com``（实测 HTTP 200）。
2. 本机 ``~/.cache/huggingface`` 已被 165GB 其它模型占用，为避免污染，
   本平台把 HF 缓存重定向到 ``<项目>/models/hf_cache``。
3. 企业内网离线部署时，先在有网机器执行本脚本，再把 ``models/hf_cache``
   整体拷贝到内网机器即可。

用法::

    python scripts/download_esm2.py                    # 下载主模型
    python scripts/download_esm2.py --fallback         # 额外下载 CPU 降级小模型
    python scripts/download_esm2.py --model facebook/esm2_t12_35M_UR50D
    python scripts/download_esm2.py --list             # 仅检查本地状态
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _bootstrap_env() -> tuple[str, Path]:
    """在导入 huggingface_hub 之前设置 HF_ENDPOINT / HF_HOME。"""
    from backend.app.core.config import get_settings

    settings = get_settings()
    hf_home = Path(settings.models_dir) / "hf_cache"
    hf_home.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
    # 关闭遥测与软链接告警噪声
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return settings.hf_endpoint, hf_home


def _is_cached(repo_id: str, hf_home: Path) -> tuple[bool, float]:
    """检查模型是否已在本地缓存中（不触发下载）。"""
    slug = "models--" + repo_id.replace("/", "--")
    model_dir = hf_home / "hub" / slug
    if not model_dir.exists():
        return False, 0.0
    total = 0
    for item in model_dir.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    # 权重文件通常大于 100MB；小于则视为不完整
    return total > 100 * 1024 * 1024, total / 1024 / 1024


def download(repo_id: str, hf_home: Path, endpoint: str) -> bool:
    """下载（或校验）一个模型仓库。"""
    cached, size_mb = _is_cached(repo_id, hf_home)
    if cached:
        print(f"[跳过] {repo_id} 已缓存（{size_mb:.1f} MB）")
        return True

    print(f"[下载] {repo_id}")
    print(f"       镜像源 : {endpoint}")
    print(f"       缓存到 : {hf_home / 'hub'}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("错误: 未安装 huggingface_hub，请先执行 pip install -r requirements.txt")
        return False

    started = time.time()
    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=str(hf_home / "hub"),
            endpoint=endpoint,
            # 只取推理需要的文件，跳过 TF/Flax 权重
            ignore_patterns=["*.h5", "*.msgpack", "*.ot", "tf_model.h5", "flax_model.msgpack"],
            max_workers=4,
        )
    except Exception as exc:
        print(f"错误: 下载 {repo_id} 失败 -> {exc}")
        print("       排查建议：")
        print("         1) 确认能访问镜像源: curl -I https://hf-mirror.com")
        print("         2) 或指定其它端点: HF_ENDPOINT=https://hf-mirror.com python scripts/download_esm2.py")
        print("         3) 内网环境请拷贝整个 models/hf_cache 目录")
        return False

    elapsed = time.time() - started
    _, size_mb = _is_cached(repo_id, hf_home)
    print(f"[完成] {repo_id}  {size_mb:.1f} MB  耗时 {elapsed:.1f}s")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 ESM-2 权重到项目独立缓存")
    parser.add_argument("--model", default=None, help="指定模型 repo id，默认取配置 esm_model")
    parser.add_argument("--fallback", action="store_true", help="同时下载 CPU 降级小模型")
    parser.add_argument("--list", action="store_true", help="仅检查本地缓存状态，不下载")
    args = parser.parse_args()

    endpoint, hf_home = _bootstrap_env()
    from backend.app.core.config import get_settings

    settings = get_settings()

    targets: list[str] = []
    if args.model:
        targets.append(args.model)
    else:
        targets.append(settings.esm_model)
    if args.fallback:
        targets.append(settings.esm_fallback_model)

    print("=" * 70)
    print(" ESM-2 权重获取")
    print("=" * 70)
    print(f" HF_ENDPOINT : {endpoint}")
    print(f" HF_HOME     : {hf_home}")
    print("=" * 70)

    if args.list:
        for repo_id in targets:
            cached, size_mb = _is_cached(repo_id, hf_home)
            state = f"已缓存 {size_mb:.1f} MB" if cached else "未缓存"
            print(f"  {repo_id:45s} {state}")
        return 0

    ok = True
    for repo_id in targets:
        ok = download(repo_id, hf_home, endpoint) and ok

    print("=" * 70)
    if ok:
        print(" 全部就绪。可启动服务： bash run_server.sh")
    else:
        print(" 存在未完成的下载，请根据上方提示排查。")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    if not os.environ.get("PYTHONNOUSERSITE"):
        print("提示: 建议设置 PYTHONNOUSERSITE=1，避免 ~/.local 用户包污染 conda 环境。")
    raise SystemExit(main())
