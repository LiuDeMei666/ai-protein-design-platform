#!/usr/bin/env python
"""初始化演示数据。

设计原则
--------
**只写入真实来源的数据**：三类标准蛋白序列取自 UniProt（由
``fetch_seed_sequences.py`` 预先拉取），项目与备注中显式标注为演示数据。

**不写入任何合成实验记录**。实验数据（实测值）必须来自企业真实实验——
写入伪造的"实测值"会被误当成真实结果参与对比分析与模型训练，
这是不可接受的。因此本脚本只为演示目的创建序列记录，实验记录留空，
等待企业按《平台使用手册》导入。

用法::

    python scripts/seed_demo_data.py            # 初始化演示数据
    python scripts/seed_demo_data.py --reset    # 先清空演示项目再重建
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SEEDS_DIR = PROJECT_ROOT / "data" / "seeds"

DEMO_PROJECT_NAME = "演示项目（示例数据）"
DEMO_PROJECT_NOTE = (
    "平台初始化脚本创建。包含三类标准蛋白的参考序列（来源：UniProt），"
    "用于快速体验流程。所有内容均为公开参考数据，不含任何企业实验数据。"
)

#: (种子文件, 记录名称, 蛋白类型, 备注)
SEEDS: tuple[tuple[str, str, str, str], ...] = (
    (
        "collagen_col1a1_human.fasta",
        "人 I 型胶原 α1 链 (COL1A1, P02452)",
        "collagen",
        "UniProt P02452，1464 aa 前体。三股螺旋域约从第 179 位开始。",
    ),
    (
        "protease_subtilisin.fasta",
        "枯草杆菌蛋白酶 BPN' (P00782)",
        "protease",
        "UniProt P00782，382 aa 前体（成熟酶 275 aa，前导肽 1-107 位）。",
    ),
    (
        "protein_a_staph.fasta",
        "金黄色葡萄球菌蛋白 A (P38507)",
        "protein_a",
        "UniProt P38507，508 aa。含串联 Ig 结合结构域。",
    ),
)


def load_fasta(path: Path) -> tuple[str, str]:
    """读取 FASTA，返回 (header, sequence)。"""
    header = ""
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            header = header or line[1:].strip()
        elif line.strip():
            chunks.append(line.strip())
    return header, "".join(chunks).upper()


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化平台演示数据")
    parser.add_argument("--reset", action="store_true", help="先删除同名演示项目及其数据")
    args = parser.parse_args()

    from backend.app.core.logging import setup_logging
    from backend.app.db.base import session_scope
    from backend.app.db.init_db import init_db
    from backend.app.db.models import ExperimentRecord, Project, ProteinSequence
    from backend.app.services.sequence.validator import validate_sequence

    setup_logging()
    init_db()

    print("=" * 78)
    print(" 初始化演示数据")
    print("=" * 78)

    with session_scope() as session:
        existing = (
            session.query(Project).filter(Project.name == DEMO_PROJECT_NAME).one_or_none()
        )
        if existing is not None and args.reset:
            session.query(ExperimentRecord).filter(
                ExperimentRecord.project_id == existing.id
            ).delete()
            session.delete(existing)
            session.flush()
            print(f" 已清除旧的演示项目（含其序列与实验记录）")

        project = (
            session.query(Project).filter(Project.name == DEMO_PROJECT_NAME).one_or_none()
        )
        if project is None:
            project = Project(
                name=DEMO_PROJECT_NAME,
                description=DEMO_PROJECT_NOTE,
                host_system="ecoli",
            )
            session.add(project)
            session.flush()
            print(f" 已创建演示项目：{DEMO_PROJECT_NAME}（id={project.id}）")
        else:
            print(f" 演示项目已存在：{DEMO_PROJECT_NAME}（id={project.id}）")

        inserted = 0
        skipped = 0
        for filename, name, protein_type, note in SEEDS:
            path = SEEDS_DIR / filename
            if not path.exists():
                print(f"  [跳过] 缺少种子文件 {filename}，请先执行 fetch_seed_sequences.py")
                skipped += 1
                continue

            _header, sequence = load_fasta(path)
            check = validate_sequence(sequence)
            if not check.ok:
                print(f"  [跳过] {filename} 校验失败：{check.errors}")
                skipped += 1
                continue

            duplicate = (
                session.query(ProteinSequence)
                .filter(
                    ProteinSequence.project_id == project.id,
                    ProteinSequence.sha256 == check.sha256,
                )
                .one_or_none()
            )
            if duplicate is not None:
                print(f"  [已存在] {name}")
                skipped += 1
                continue

            session.add(
                ProteinSequence(
                    project_id=project.id,
                    name=name,
                    protein_type=protein_type,
                    sequence=check.sequence,
                    length=check.length,
                    sha256=check.sha256,
                    note=note,
                )
            )
            session.flush()
            print(f"  [新增] {name}（{check.length} aa，类型 {protein_type}）")
            inserted += 1

    print()
    print(" 说明：**未写入任何实验记录**。")
    print("       实验数据必须来自真实实验；平台不会用合成数据冒充实测值。")
    print("       导入方式见《平台使用手册》第 2.5 节。")
    print("=" * 78)
    print(f" 结果：新增 {inserted} 条序列，跳过 {skipped} 条")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
