#!/usr/bin/env python
"""拉取宿主密码子使用表（Kazusa Codon Usage Database）。

用途
----
表达量趋势预测需要**真实**的宿主密码子偏好数据来计算密码子适应指数（CAI）。
本项目不使用任何凭记忆填写或估算的密码子频率——全部从 Kazusa 官方数据库拉取，
并把"参考集规模"一并记录，让使用者能判断该表的可信度。

已确认的条目（2026-09 实测）
---------------------------
======================  ==========  ==========  ==============
宿主                    species ID  CDS 数      密码子数
======================  ==========  ==========  ==============
E. coli W3110 (K-12)    316407      4332        1,372,057
Pichia pastoris         4922        137         81,301
Bacillus subtilis       1423        2529        815,445
Cricetulus griseus(CHO) 10029       331         153,527
Saccharomyces cerevisiae 4932       14411       6,534,504
======================  ==========  ==========  ==============

输出: ``data/seeds/codon_usage/<table>.json``

用法::

    python scripts/fetch_codon_usage.py           # 拉取全部宿主
    python scripts/fetch_codon_usage.py --list    # 查看本地已有表
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "data" / "seeds" / "codon_usage"
KAZUSA_URL = "https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi"

#: table 名 -> (Kazusa species id, 宿主显示名)
SPECIES: dict[str, tuple[int, str]] = {
    "ecoli_k12": (316407, "Escherichia coli W3110 (K-12)"),
    "pichia_pastoris": (4922, "Pichia pastoris"),
    "bacillus_subtilis": (1423, "Bacillus subtilis"),
    "cho": (10029, "Cricetulus griseus (CHO)"),
    "saccharomyces_cerevisiae": (4932, "Saccharomyces cerevisiae"),
}

_ROW_PATTERN = re.compile(
    r"([ACGTU]{3})\s+([A-Z*])\s+([\d.]+)\s+([\d.]+)\s*\(\s*(\d+)\)"
)
_HEADER_PATTERN = re.compile(r"<i>([^<]+)</i>\[[a-z]*\]:\s*(\d+)\s*CDS's\s*\((\d+)\s*codons\)")


def fetch_table(species_id: int, retries: int = 3) -> tuple[str, int, int, dict]:
    """从 Kazusa 拉取一个物种的密码子表。

    Returns:
        (表头物种名, CDS 数, 密码子总数, {DNA密码子: 记录})
    """
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(timeout=40.0, follow_redirects=True) as client:
                response = client.get(
                    KAZUSA_URL,
                    params={"species": species_id, "aa": 1, "style": "N"},
                )
                response.raise_for_status()
                html = response.text

            header_match = _HEADER_PATTERN.search(html)
            if header_match is None:
                raise ValueError(f"未能解析表头，species={species_id} 可能不存在")
            organism = header_match.group(1).strip()
            n_cds = int(header_match.group(2))
            n_codons = int(header_match.group(3))

            codons: dict[str, dict] = {}
            for match in _ROW_PATTERN.finditer(html):
                rna_codon, amino_acid, fraction, per_thousand, count = match.groups()
                # Kazusa 输出 RNA 形式（UUU），统一转为 DNA 形式（TTT）
                dna_codon = rna_codon.replace("U", "T")
                codons[dna_codon] = {
                    "amino_acid": amino_acid,
                    "fraction": float(fraction),
                    "per_thousand": float(per_thousand),
                    "count": int(count),
                }

            if len(codons) < 60:
                raise ValueError(f"只解析到 {len(codons)} 个密码子，数据可能不完整")

            return organism, n_cds, n_codons, codons

        except Exception as exc:
            last_error = exc
            print(f"    第 {attempt} 次尝试失败: {exc}")
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"拉取 species={species_id} 失败: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="拉取宿主密码子使用表")
    parser.add_argument("--list", action="store_true", help="仅列出本地已缓存的表")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        print(f"密码子表目录: {OUTPUT_DIR}")
        for path in sorted(OUTPUT_DIR.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            print(
                f"  {path.name:32s} {payload.get('organism', ''):35s} "
                f"{payload.get('n_cds', 0):>6d} CDS / {payload.get('n_codons', 0):>9d} codons"
            )
        return 0

    print("=" * 78)
    print(" 拉取宿主密码子使用表（Kazusa Codon Usage Database）")
    print("=" * 78)

    failed: list[str] = []
    for table_name, (species_id, label) in SPECIES.items():
        print(f"\n[{table_name}] {label} (species={species_id})")
        try:
            organism, n_cds, n_codons, codons = fetch_table(species_id)
        except Exception as exc:
            print(f"  失败: {exc}")
            failed.append(table_name)
            continue

        payload = {
            "table_name": table_name,
            "label": label,
            "organism": organism,
            "species_id": species_id,
            "n_cds": n_cds,
            "n_codons": n_codons,
            "source": f"{KAZUSA_URL}?species={species_id}",
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "codons": codons,
        }
        path = OUTPUT_DIR / f"{table_name}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  物种  : {organism}")
        print(f"  规模  : {n_cds} CDS / {n_codons} codons（样本越大越可信）")
        print(f"  已保存: {path.name}")

    print("\n" + "=" * 78)
    if failed:
        print(f" 有 {len(failed)} 个表未获取: {'、'.join(failed)}")
        print(" 表达量预测将对这些宿主降级为'仅使用不依赖密码子表的指标'。")
        return 1
    print(" 全部密码子表就绪。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
