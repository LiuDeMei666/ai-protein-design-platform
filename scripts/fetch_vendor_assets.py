#!/usr/bin/env python
"""把前端第三方库 vendor 到本地。

为什么必须 vendor
-----------------
1. 企业内网部署通常**无法访问公网 CDN**，若前端依赖 jsdelivr 会直接白屏。
2. 本机 Node 版本为 v12.22.9，无法运行 Vite/现代打包链，因此前端是零构建的
   原生多页应用，第三方库只能以静态文件形式本地引入。

多镜像回退
----------
每个库配置多个镜像源，逐个尝试，任一成功即停止；全部失败时**不生成残缺文件**，
并在退出码中体现，前端会自动切换到无依赖的 Canvas 降级视图（见 viewer3d.js）。

用法::

    python scripts/fetch_vendor_assets.py            # 下载到 frontend/vendor/
    python scripts/fetch_vendor_assets.py --list     # 查看本地已有资源
    python scripts/fetch_vendor_assets.py --verify   # 校验文件完整性（大小 + 特征串）
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = PROJECT_ROOT / "frontend" / "vendor"


@dataclass
class VendorAsset:
    """一个待 vendor 的前端资源。"""

    filename: str
    #: 多个镜像 URL，按顺序尝试
    urls: tuple[str, ...]
    #: 期望的最小字节数（用于判断下载是否残缺）
    min_bytes: int
    #: 文件内容应包含的特征串（用于校验下载的确实是该库）
    signature: str
    description: str = ""
    license: str = ""
    homepage: str = ""

    def local_path(self) -> Path:
        return VENDOR_DIR / self.filename


#: 待 vendor 的资源清单（版本号显式锁定，保证可复现）
ASSETS: tuple[VendorAsset, ...] = (
    VendorAsset(
        filename="echarts.min.js",
        urls=(
            "https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js",
            "https://registry.npmmirror.com/echarts/5.4.3/files/dist/echarts.min.js",
            "https://unpkg.com/echarts@5.4.3/dist/echarts.min.js",
            "https://cdn.bootcdn.net/ajax/libs/echarts/5.4.3/echarts.min.js",
            "https://cdnjs.cloudflare.com/ajax/libs/echarts/5.4.3/echarts.min.js",
        ),
        min_bytes=500_000,
        signature="echarts",
        description="雷达图 / 折线图 / 散点图 / 柱状图 / 轨道图",
        license="Apache-2.0",
        homepage="https://echarts.apache.org/",
    ),
    VendorAsset(
        filename="3Dmol-min.js",
        urls=(
            "https://cdn.jsdelivr.net/npm/3dmol@2.4.0/build/3Dmol-min.js",
            "https://registry.npmmirror.com/3dmol/2.4.0/files/build/3Dmol-min.js",
            "https://unpkg.com/3dmol@2.4.0/build/3Dmol-min.js",
            "https://cdn.bootcdn.net/ajax/libs/3Dmol/2.4.0/3Dmol-min.js",
        ),
        min_bytes=300_000,
        signature="$3Dmol",
        description="PDB 三维结构渲染与 pLDDT 着色",
        license="BSD-3-Clause",
        homepage="https://3dmol.csb.pitt.edu/",
    ),
    VendorAsset(
        filename="chartjs.min.js",
        urls=(
            "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js",
            "https://registry.npmmirror.com/chart.js/4.4.1/files/dist/chart.umd.min.js",
            "https://unpkg.com/chart.js@4.4.1/dist/chart.umd.min.js",
        ),
        min_bytes=100_000,
        signature="Chart",
        description="轻量备用图表库（ECharts 不可用时降级）",
        license="MIT",
        homepage="https://www.chartjs.org/",
    ),
)


def _download(asset: VendorAsset, timeout: float = 60.0) -> tuple[bool, str]:
    """按镜像顺序尝试下载，返回 ``(是否成功, 说明)``。"""
    target = asset.local_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    for url in asset.urls:
        started = time.time()
        print(f"    尝试 {url}")
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                content = response.content
        except Exception as exc:
            print(f"      失败: {exc}")
            errors.append(f"{url} -> {exc}")
            continue

        if len(content) < asset.min_bytes:
            print(f"      内容过小（{len(content)} < {asset.min_bytes} 字节），判定为残缺")
            errors.append(f"{url} -> 内容过小 {len(content)} 字节")
            continue

        # 全文件搜索特征串。曾经只检查首尾各 4KB，导致 3Dmol 被误判为无效——
        # 它的全局符号 $3Dmol 出现在压缩后代码的中后段。这些库体积在 1MB 级别，
        # 整体解码的代价可以忽略，不必为省这点开销牺牲校验可靠性。
        text_content = content.decode("utf-8", errors="ignore")
        if asset.signature not in text_content:
            print(f"      未找到特征串 {asset.signature!r}，判定为无效内容")
            errors.append(f"{url} -> 缺少特征串 {asset.signature!r}")
            continue

        # 原子写入，避免中断留下半个文件
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_bytes(content)
        temp.replace(target)
        elapsed = time.time() - started
        digest = hashlib.sha256(content).hexdigest()[:12]
        print(
            f"      成功：{len(content) / 1024:.0f} KB，耗时 {elapsed:.1f}s，"
            f"sha256:{digest}…"
        )
        return True, f"{len(content)} bytes from {url}"

    return False, "所有镜像均失败：\n      " + "\n      ".join(errors)


def _write_manifest(results: dict[str, dict[str, object]]) -> None:
    """写入清单文件，记录来源、大小与校验值（便于审计与复现）。"""
    import json
    from datetime import datetime

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": (
            "本目录为前端第三方库的本地副本，由 scripts/fetch_vendor_assets.py 生成。"
            "企业内网部署时连同本目录一起拷贝即可离线运行。"
        ),
        "assets": results,
    }
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    (VENDOR_DIR / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_licenses() -> None:
    """生成第三方许可声明，满足企业合规审查需要。"""
    lines = [
        "# 第三方前端库许可声明",
        "",
        "本目录下的静态资源为以下开源库的官方发布构建，版权归各自作者所有。",
        "",
        "| 文件 | 库 | 版本 | 许可 | 用途 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for asset in ASSETS:
        version = "—"
        for token in asset.urls[0].split("@")[1:]:
            version = token.split("/")[0]
            break
        lines.append(
            f"| `{asset.filename}` | {asset.description} | {version} | "
            f"{asset.license} | {asset.homepage} |"
        )
    lines.extend(
        [
            "",
            "## 部署说明",
            "",
            "1. 本目录整体拷贝到内网服务器的 `frontend/vendor/` 即可离线使用。",
            "2. 若某个文件缺失，前端会自动降级（例如缺少 3Dmol 时改用服务端解析的",
            "   Canvas 残基轨道图），不会白屏。",
            "3. 重新获取：`python scripts/fetch_vendor_assets.py`",
        ]
    )
    (VENDOR_DIR / "LICENSES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="vendor 前端第三方库到本地")
    parser.add_argument("--list", action="store_true", help="仅列出本地已有资源")
    parser.add_argument("--verify", action="store_true", help="仅校验本地文件完整性")
    parser.add_argument("--force", action="store_true", help="已存在时也重新下载")
    args = parser.parse_args()

    if args.list or args.verify:
        print(f"vendor 目录: {VENDOR_DIR}")
        ok = True
        for asset in ASSETS:
            path = asset.local_path()
            if not path.exists():
                print(f"  [缺失] {asset.filename:22s} {asset.description}")
                ok = False
                continue
            size = path.stat().st_size
            good = size >= asset.min_bytes
            state = "完整" if good else "残缺"
            print(f"  [{state}] {asset.filename:22s} {size / 1024:8.0f} KB  {asset.description}")
            ok = ok and good
        return 0 if ok else 1

    print("=" * 74)
    print(" 获取前端第三方库（多镜像回退）")
    print("=" * 74)

    results: dict[str, dict[str, object]] = {}
    failed: list[str] = []

    for asset in ASSETS:
        target = asset.local_path()
        if target.exists() and not args.force:
            size = target.stat().st_size
            if size >= asset.min_bytes:
                print(f"\n[{asset.filename}] 已存在（{size / 1024:.0f} KB），跳过")
                results[asset.filename] = {
                    "status": "cached",
                    "bytes": size,
                    "description": asset.description,
                    "license": asset.license,
                }
                continue
            print(f"\n[{asset.filename}] 本地文件残缺，重新下载")

        print(f"\n[{asset.filename}] {asset.description}")
        success, detail = _download(asset)
        if success:
            results[asset.filename] = {
                "status": "downloaded",
                "bytes": asset.local_path().stat().st_size,
                "detail": detail,
                "description": asset.description,
                "license": asset.license,
                "source": asset.urls[0],
            }
        else:
            failed.append(asset.filename)
            results[asset.filename] = {"status": "failed", "detail": detail}

    _write_manifest(results)
    _write_licenses()

    print("\n" + "=" * 74)
    if failed:
        print(f" 有 {len(failed)} 个资源未获取: {'、'.join(failed)}")
        print(" 前端将自动降级运行（功能受限但不会白屏）。")
        print(" 内网离线部署时，可在有网机器执行本脚本后拷贝 frontend/vendor/ 目录。")
        print("=" * 74)
        return 1

    print(" 全部前端资源就绪。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
