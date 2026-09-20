#!/usr/bin/env python
"""拉取标准化测试案例所需的标准序列（UniProt）。

覆盖需求文档点名的三类目标蛋白：

============  ============================================================
类型           对象
============  ============================================================
胶原蛋白     人 COL1A1（I 型胶原 α1 链），三股螺旋区富含 Gly-X-Y
重组蛋白酶    枯草杆菌蛋白酶（Subtilisin），工业碱性蛋白酶代表
蛋白 A       金黄色葡萄球菌 Protein A（spa），含 5 个 Ig 结合结构域
============  ============================================================

所有序列来自 UniProt 官方 REST API，**不内置任何人工编造的序列**。
拉取结果落盘到 ``data/seeds/*.fasta`` 后即可离线复用。

用法::

    python scripts/fetch_seed_sequences.py          # 拉取并保存
    python scripts/fetch_seed_sequences.py --list    # 只查看本地已有文件
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SEEDS_DIR = PROJECT_ROOT / "data" / "seeds"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

#: (输出文件名, 说明, UniProt 查询串)
TARGETS: list[tuple[str, str, str]] = [
    (
        "collagen_col1a1_human.fasta",
        "人 I 型胶原 α1 链 (COL1A1)",
        '(gene:COL1A1) AND (organism_id:9606) AND (reviewed:true)',
    ),
    (
        "protease_subtilisin.fasta",
        "枯草杆菌蛋白酶 (Subtilisin)",
        '(protein_name:"Subtilisin") AND (reviewed:true) AND (taxonomy_id:1386)',
    ),
    (
        "protein_a_staph.fasta",
        "金黄色葡萄球菌蛋白 A (Protein A / spa)",
        '(gene:spa) AND (organism_id:1280) AND (reviewed:true)',
    ),
]


def fetch_one(query: str, retries: int = 3) -> tuple[str, dict]:
    """按查询串取第一条 reviewed 条目的 FASTA 与元数据。"""
    params = {
        "query": query,
        "format": "fasta",
        "size": 1,
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(UNIPROT_SEARCH, params=params)
                response.raise_for_status()
                fasta = response.text.strip()
                if not fasta.startswith(">"):
                    raise ValueError(f"返回内容不是 FASTA: {fasta[:120]}")

                # 取 JSON 元数据（accession / 长度 / 名称）
                meta_response = client.get(
                    UNIPROT_SEARCH,
                    params={
                        "query": query,
                        "format": "json",
                        "size": 1,
                        "fields": "accession,id,protein_name,length,organism_name",
                    },
                )
                meta_response.raise_for_status()
                payload = meta_response.json()
                entry = (payload.get("results") or [{}])[0]
                meta = {
                    "accession": entry.get("primaryAccession", ""),
                    "entry_name": entry.get("uniProtkbId", ""),
                    "protein_name": (
                        entry.get("proteinDescription", {})
                        .get("recommendedName", {})
                        .get("fullName", {})
                        .get("value", "")
                    ),
                    "organism": entry.get("organism", {}).get("scientificName", ""),
                    "length": entry.get("sequence", {}).get("length", 0),
                }
                return fasta, meta
        except Exception as exc:
            last_error = exc
            print(f"    第 {attempt} 次尝试失败: {exc}")
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"UniProt 查询失败: {query} -> {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="拉取标准化测试案例序列")
    parser.add_argument("--list", action="store_true", help="仅列出本地已有文件")
    args = parser.parse_args()

    SEEDS_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        print(f"种子序列目录: {SEEDS_DIR}")
        for path in sorted(SEEDS_DIR.glob("*.fasta")):
            text = path.read_text(encoding="utf-8")
            header = text.splitlines()[0] if text else ""
            print(f"  {path.name:38s} {header[:70]}")
        return 0

    print("=" * 74)
    print(" 拉取标准化测试案例序列（UniProt REST）")
    print("=" * 74)

    manifest: dict[str, dict] = {}
    failed: list[str] = []

    for filename, label, query in TARGETS:
        print(f"\n[{label}]")
        print(f"  查询: {query}")
        try:
            fasta, meta = fetch_one(query)
        except Exception as exc:
            print(f"  失败: {exc}")
            failed.append(label)
            continue

        path = SEEDS_DIR / filename
        path.write_text(fasta + "\n", encoding="utf-8")
        sequence = "".join(
            line.strip() for line in fasta.splitlines() if not line.startswith(">")
        )
        print(f"  已保存: {path.name}")
        print(f"  条目  : {meta['accession']} {meta['entry_name']}")
        print(f"  名称  : {meta['protein_name']}")
        print(f"  物种  : {meta['organism']}")
        print(f"  长度  : {len(sequence)} aa")
        manifest[filename] = {**meta, "length_actual": len(sequence), "label": label}

    manifest_path = SEEDS_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n清单已写入: {manifest_path}")

    print("=" * 74)
    if failed:
        print(f" 有 {len(failed)} 项未获取: {'、'.join(failed)}")
        print(" 请检查网络，或从 UniProt 官网手工下载后放入 data/seeds/")
        return 1
    print(" 全部序列就绪。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
