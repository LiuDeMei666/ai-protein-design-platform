#!/usr/bin/env python
"""把 Word（.docx）文档反向转换为 Markdown。

为什么需要这个脚本
------------------
项目里 ``scripts/make_docs.py`` 负责 md -> docx（自行编写交付文档）。
但合作方回传的需求文档只有 .docx，且本机**没有安装 pandoc**，需要反向通道
把它还原成 Markdown，才能纳入版本管理与全文检索。

识别的结构
----------
- **标题**：优先读 Word 内置标题样式（``Heading N`` / ``标题 N``）；没有样式时
  （合作方文档常常整篇都是 Normal）退化为三级启发式：
  中文章节编号（``一、二、``）→ 2 级；整段加粗的 ``1. xxx`` → 3 级；
  全文最大字号且位于开头 → 1 级。
- **列表**：读 ``w:numPr`` 的 numId/ilvl，并解析 ``numbering.xml`` 判断
  有序（decimal 等）还是无序（bullet）。序号按各 numId 独立计数。
- **表格**：转为 Markdown 表格，自动补齐列数并转义 ``|``。
- **行内**：加粗、斜体、等宽代码（按字体名判断）。

用法::

    python scripts/convert_docx_to_md.py docs/AI辅助蛋白设计平台开发需求简介2.docx
    python scripts/convert_docx_to_md.py docs/xxx.docx -o docs/xxx.md

不指定 ``-o`` 时，输出到与输入同目录的同名 ``.md`` 文件。
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterator

from docx import Document
from docx.document import Document as DocumentObject
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Word 内置标题样式名 -> Markdown 标题级别
HEADING_STYLE = re.compile(r"^(?:Heading|标题)\s*(\d+)$", re.IGNORECASE)

#: 中文章节编号，如 ``一、项目背景``、``（一）总体目标``
CN_SECTION = re.compile(r"^(?:第?[一二三四五六七八九十百]+[、.．]|[（(][一二三四五六七八九十]+[）)])\s*\S")

#: 阿拉伯数字编号，如 ``1. 蛋白结构与性质预测模块``
NUM_SECTION = re.compile(r"^\d+\s*[.、．)]\s*\S")

#: 等宽字体名前缀（判定行内代码）
MONO_FONTS = ("Consolas", "Courier", "Monaco", "Menlo", "Source Code", "DejaVu Sans Mono")

#: 有序列表的 numbering.xml numFmt 取值
ORDERED_FORMATS = {
    "decimal", "decimalZero", "lowerLetter", "upperLetter",
    "lowerRoman", "upperRoman", "chineseCounting", "japaneseCounting",
    "ideographDigital", "koreanCounting",
}

#: 视为"正文空行"的最小非空判断
MAX_TITLE_LENGTH = 60
MAX_BOLD_HEADING_LENGTH = 40


# --------------------------------------------------------------------------- #
# 正文元素按文档顺序遍历
# --------------------------------------------------------------------------- #
def iter_block_items(document: DocumentObject) -> Iterator[Paragraph | Table]:
    """按文档中的**真实先后顺序**产出段落与表格。

    ``document.paragraphs`` 会把所有段落排在所有表格之前，顺序与阅读顺序不符。
    这里直接遍历 body 的直接子元素，对每个元素包一层 python-docx 对象。
    """
    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            yield Paragraph(child, document)
        elif tag == "tbl":
            yield Table(child, document)


# --------------------------------------------------------------------------- #
# 列表信息
# --------------------------------------------------------------------------- #
def read_numbering(docx_path: Path) -> dict[tuple[int, int], bool]:
    """解析 ``word/numbering.xml``，返回 ``{(numId, ilvl): 是否有序}``。

    为什么必须读真实定义而不是猜
    ---------------------------
    ``w:numPr`` 只给出 numId，**编号样式与符号完全由 numbering.xml 决定**。
    同一个 numId 在有的文档里是圆点、有的文档里是 ``1)``，凭 numId 猜会猜错。
    """
    formats: dict[tuple[int, int], bool] = {}
    try:
        with zipfile.ZipFile(docx_path) as archive:
            xml = archive.read("word/numbering.xml").decode("utf-8")
    except (KeyError, FileNotFoundError):
        return formats

    # abstractNumId -> {ilvl: numFmt}
    abstract: dict[str, dict[int, str]] = {}
    for block in re.findall(r"<w:abstractNum\b.*?</w:abstractNum>", xml, re.S):
        match = re.search(r'w:abstractNumId="(\d+)"', block)
        if not match:
            continue
        levels: dict[int, str] = {}
        for level in re.findall(r"<w:lvl\b.*?</w:lvl>", block, re.S):
            ilvl = re.search(r'w:ilvl="(\d+)"', level)
            fmt = re.search(r'<w:numFmt w:val="([^"]+)"', level)
            if ilvl and fmt:
                levels[int(ilvl.group(1))] = fmt.group(1)
        abstract[match.group(1)] = levels

    # numId -> abstractNumId
    for block in re.findall(r'<w:num w:numId="\d+"[^>]*>.*?</w:num>', xml, re.S):
        num_id = re.search(r'w:numId="(\d+)"', block)
        abs_id = re.search(r'<w:abstractNumId w:val="(\d+)"', block)
        if not (num_id and abs_id):
            continue
        levels = abstract.get(abs_id.group(1), {})
        for ilvl, fmt in levels.items():
            formats[(int(num_id.group(1)), ilvl)] = fmt in ORDERED_FORMATS
    return formats


def list_info(paragraph: Paragraph) -> tuple[int, int] | None:
    """返回 ``(numId, ilvl)``；非列表段落返回 ``None``。"""
    properties = paragraph._p.pPr
    if properties is None or properties.numPr is None:
        return None
    num_id = properties.numPr.numId
    ilvl = properties.numPr.ilvl
    return (
        num_id.val if num_id is not None else 0,
        ilvl.val if ilvl is not None else 0,
    )


def indent_level(paragraph: Paragraph) -> int:
    """由 ``w:ind`` 的 left 缩进推断嵌套层级（列表未标 ilvl 时兜底）。

    Word 的缩进单位是 twips（1/20 pt）。正文左缩进通常 0，一级列表约 360-720，
    这里以 360 twips 为一个层级，仅作粗略兜底。
    """
    properties = paragraph._p.pPr
    if properties is None or properties.ind is None or properties.ind.left is None:
        return 0
    return max(0, int(properties.ind.left) // 360)


# --------------------------------------------------------------------------- #
# 标题识别
# --------------------------------------------------------------------------- #
def paragraph_font_size(paragraph: Paragraph) -> float | None:
    """取段落中最大的字号（pt）。"""
    sizes = [run.font.size.pt for run in paragraph.runs if run.font.size is not None]
    return max(sizes) if sizes else None


def is_all_bold(paragraph: Paragraph) -> bool:
    """段落（所有非空白 run）是否整体加粗。"""
    runs = [run for run in paragraph.runs if run.text.strip()]
    return bool(runs) and all(bool(run.bold) for run in runs)


def detect_heading(
    paragraph: Paragraph,
    *,
    is_first: bool,
    max_font_size: float | None,
) -> int | None:
    """返回标题级别（1-6），非标题返回 ``None``。

    判定顺序即优先级：显式样式 > 章节编号 > 整段加粗编号 > 首段最大字号。
    """
    text = paragraph.text.strip()
    if not text:
        return None

    # 1) 显式标题样式最可靠
    match = HEADING_STYLE.match(paragraph.style.name.strip())
    if match:
        return min(int(match.group(1)), 6)

    # 2) 中文章节编号（一、二、三、）—— 合作方文档最常见的顶层分节
    if CN_SECTION.match(text) and len(text) <= MAX_TITLE_LENGTH:
        return 2

    # 3) 整段加粗的 "1. xxx" —— 二级分节
    if (
        NUM_SECTION.match(text)
        and len(text) <= MAX_BOLD_HEADING_LENGTH
        and is_all_bold(paragraph)
    ):
        return 3

    # 4) 开头处字号最大的短句视为文档标题
    if is_first and max_font_size is not None and len(text) <= MAX_TITLE_LENGTH:
        size = paragraph_font_size(paragraph)
        if size is not None and size >= max_font_size:
            return 1

    return None


def document_max_font_size(document: DocumentObject) -> float | None:
    """全文最大字号，用于识别无样式的文档标题。"""
    sizes: list[float] = []
    for paragraph in document.paragraphs:
        size = paragraph_font_size(paragraph)
        if size is not None and paragraph.text.strip():
            sizes.append(size)
    return max(sizes) if sizes else None


# --------------------------------------------------------------------------- #
# 行内与表格渲染
# --------------------------------------------------------------------------- #
def wrap_inline(text: str, marker: str) -> str:
    """给文本套行内标记，**保留首尾空白在标记之外**。

    Markdown 的 ``**bold**`` 分隔符两侧不能贴着空白，否则不生效。
    Word 里一个词常被拆成多个 run，空白恰好落在加粗 run 末尾，直接包裹会失效。
    """
    if not text.strip():
        return text
    leading = text[: len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()) :]
    return f"{leading}{marker}{text.strip()}{marker}{trailing}"


def render_inline(paragraph: Paragraph) -> str:
    """渲染段落的行内文本（加粗 / 斜体 / 代码）。"""
    pieces: list[list] = []
    for run in paragraph.runs:
        if not run.text:
            continue
        font_name = run.font.name or ""
        is_code = font_name.startswith(MONO_FONTS)
        entry = [run.text, bool(run.bold), bool(run.italic), is_code]
        # 合并相邻同样式的片段，避免产出 **a****b** 这类碎片
        if pieces and pieces[-1][1:] == entry[1:]:
            pieces[-1][0] += entry[0]
        else:
            pieces.append(entry)

    rendered: list[str] = []
    for text, bold, italic, is_code in pieces:
        if is_code:
            rendered.append(wrap_inline(text, "`"))
        elif bold and italic:
            rendered.append(wrap_inline(text, "***"))
        elif bold:
            rendered.append(wrap_inline(text, "**"))
        elif italic:
            rendered.append(wrap_inline(text, "*"))
        else:
            rendered.append(text)
    return "".join(rendered).rstrip()


def escape_cell(text: str) -> str:
    """表格单元格转义：竖线会截断表格，换行需转 ``<br>``。"""
    return text.replace("|", r"\|").replace("\n", "<br>")


def render_table(table: Table) -> str:
    """把 Word 表格转为 Markdown 表格。"""
    rows: list[list[str]] = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            # 单元格内多段落用 <br> 连接
            pieces = [render_inline(p).strip() for p in cell.paragraphs]
            cells.append(escape_cell("<br>".join(piece for piece in pieces if piece)))
        rows.append(cells)

    if not rows:
        return ""

    width = max(len(row) for row in rows)
    for row in rows:
        row.extend([""] * (width - len(row)))

    header, *body = rows
    # 全空表头会导致 Markdown 表格结构无效，兜底补列名
    header = [cell or f"列{index + 1}" for index, cell in enumerate(header)]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 主转换流程
# --------------------------------------------------------------------------- #
def convert(docx_path: Path) -> str:
    """把 docx 转为 Markdown 文本。"""
    document = Document(str(docx_path))
    numbering = read_numbering(docx_path)
    max_size = document_max_font_size(document)

    blocks: list[str] = []
    counters: dict[tuple[int, int], int] = {}
    # 连续列表项要攒成**一个**块，用单换行连接。若每项之间插空行，
    # Markdown 会判定为"松散列表"并在每个条目内包一层 <p>，排版变松垮。
    pending_list: list[str] = []
    seen_content = False

    def flush_list() -> None:
        if pending_list:
            blocks.append("\n".join(pending_list))
            pending_list.clear()

    for block in iter_block_items(document):
        if isinstance(block, Table):
            flush_list()
            table_text = render_table(block)
            if table_text:
                blocks.append(table_text)
            continue

        text = render_inline(block).strip()

        # 空段落即空行，同时也是列表的分隔信号
        if not text:
            flush_list()
            continue

        # ---------- 列表 ----------
        info = list_info(block)
        if info is not None:
            num_id, ilvl = info
            ordered = numbering.get((num_id, ilvl), True)
            # 序号按 numId 独立计数：Word 里每个独立列表各有自己的 numId，
            # 被正文段落打断并不重置编号（只有显式 lvlRestart 才会）。
            counters[(num_id, ilvl)] = counters.get((num_id, ilvl), 0) + 1
            marker = f"{counters[(num_id, ilvl)]}." if ordered else "-"
            indent = "    " * (ilvl if ilvl else indent_level(block))
            pending_list.append(f"{indent}{marker} {text}")
            seen_content = True
            continue

        flush_list()

        # ---------- 标题 ----------
        level = detect_heading(block, is_first=not seen_content, max_font_size=max_size)
        if level is not None:
            # 标题直接用纯文本：原文标题整段加粗，若保留会产出 `### **xxx**`，
            # 这在 Markdown 里是无效写法（星号会被当字面量渲染出来）。
            blocks.append(f"{'#' * level} {block.text.strip()}")
            seen_content = True
            continue

        # ---------- 正文 ----------
        blocks.append(text)
        seen_content = True

    flush_list()
    return "\n\n".join(blocks) + "\n" if blocks else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把 Word（.docx）文档反向转换为 Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("docx", type=Path, help="待转换的 .docx 文件路径")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 .md 路径（默认与输入同目录同名）",
    )
    args = parser.parse_args(argv)

    source: Path = args.docx
    if not source.is_absolute():
        source = (PROJECT_ROOT / source).resolve()
    if not source.exists():
        print(f"[错误] 找不到文件：{source}", file=sys.stderr)
        return 1
    if source.suffix.lower() != ".docx":
        print(f"[错误] 仅支持 .docx，收到：{source.suffix}", file=sys.stderr)
        return 1

    target: Path = args.output if args.output else source.with_suffix(".md")
    if not target.is_absolute():
        target = (PROJECT_ROOT / target).resolve()

    markdown = convert(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")

    headings = sum(1 for line in markdown.splitlines() if line.startswith("#"))
    bullets = sum(1 for line in markdown.splitlines() if line.lstrip().startswith("- "))
    ordered = sum(1 for line in markdown.splitlines() if re.match(r"^\s*\d+\. ", line))
    tables = markdown.count("| ---")
    print(f"已转换：{source.name} -> {target}")
    print(
        f"  字符数 {len(markdown)}  |  标题 {headings}  |  "
        f"无序项 {bullets}  |  有序项 {ordered}  |  表格 {tables}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
