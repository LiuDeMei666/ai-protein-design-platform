"""序列校验、清洗与 FASTA 解析。

校验策略
--------
0. **剥离 FASTA 头**：以 ``>`` 开头的行**整行丢弃**（见 :func:`split_fasta_headers`）。
   只删 ``>`` 会让头里的字母混进序列，是比报错更隐蔽的污染。
1. 清洗：去掉空白、换行、数字、``*``（终止符）、``.`` 等非残基字符，统一大写。
2. 分级：20 种标准氨基酸为合法；``B/Z/J/X/U/O`` 为**歧义残基**——性质预测可容忍
   （会显式告警），但**突变设计必须拒绝**，因为蛋白语言模型只接受 20 字母表。
3. 绝不静默修正：所有被丢弃的字符与歧义位点都回传前端，由研发人员确认。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from ...core.config import get_settings

# 20 种标准氨基酸
STANDARD_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")
# 歧义 / 特殊残基：B(Asn|Asp) Z(Gln|Glu) J(Leu|Ile) X(任意) U(硒代半胱氨酸) O(吡咯赖氨酸)
AMBIGUOUS_AA: frozenset[str] = frozenset("BZJXUO")
AMBIGUOUS_MEANING: dict[str, str] = {
    "B": "Asn 或 Asp",
    "Z": "Gln 或 Glu",
    "J": "Leu 或 Ile",
    "X": "任意残基",
    "U": "硒代半胱氨酸",
    "O": "吡咯赖氨酸",
}

_CLEANUP_PATTERN = re.compile(r"[\s\d*.\-_|]")
_GAP_CHARS = frozenset("-.")


@dataclass
class FastaRecord:
    """FASTA 中的一条记录。"""

    header: str
    sequence: str

    @property
    def name(self) -> str:
        """取 header 第一个 token 作为名称。"""
        return self.header.split()[0] if self.header.split() else "unnamed"


@dataclass
class SequenceCheck:
    """序列校验结果。"""

    sequence: str
    length: int
    sha256: str
    invalid_chars: dict[str, int] = field(default_factory=dict)
    ambiguous_positions: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """无致命错误。"""
        return not self.errors

    @property
    def design_ready(self) -> bool:
        """是否满足突变设计对输入的要求（无歧义残基）。"""
        return self.ok and not self.ambiguous_positions

    @property
    def ambiguous_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for pos in self.ambiguous_positions:
            char = self.sequence[pos]
            counts[char] = counts.get(char, 0) + 1
        return counts


def clean_raw_sequence(raw: str) -> tuple[str, dict[str, int]]:
    """清洗原始输入，返回 (清洗后序列, 被移除字符计数)。

    仅移除明确的非残基字符（空白、数字、``*``、``-``、``.``、``|``）；
    字母保留，交由后续分级判断，避免把真正的异常暴露成静默丢弃。
    """
    removed: dict[str, int] = {}
    for match in _CLEANUP_PATTERN.finditer(raw):
        char = match.group()
        key = "whitespace" if char.isspace() else char
        removed[key] = removed.get(key, 0) + 1
    cleaned = _CLEANUP_PATTERN.sub("", raw).upper()
    return cleaned, removed


def split_fasta_headers(raw: str) -> tuple[str, list[str]]:
    """剥离 FASTA 头行，返回 ``(序列正文, 各头行内容)``。

    为什么必须**整行**丢弃，而不是只删掉 ``>``
    ------------------------------------------
    FASTA 头里含有大量字母（``sp``、``ALL5_HEVBR``、``Major``、``Hevea`` …），
    而 :data:`_CLEANUP_PATTERN` 只删除空白、数字与少数符号，**不删字母**。
    若只把 ``>`` 去掉，这些头部字母会被当作残基拼接进序列开头：

        输入  >sp|Q39967|ALL5_HEVBR Major latex allergen Hev b 5 OS=Hevea ...
              MASVEVESAATALPKNETPEVTKAEETKTEEPAAPP...

        仅删 > 后得到（实测）
              SPQALLHEVBRMAJORLATEXALLERGENHEVBOSHEVEABRASILIENSISOXMASVEVESAA...
              长度从 151 aa 变成 208 aa，凭空多出 57 个假残基

    更糟的是这些头部字母还会伪造出 ``B``/``J``/``O``/``X`` 等"歧义残基"告警
    （来自 **B**RASILIENSIS、ma**J**or、**OX**），把用户引向完全错误的方向。

    因此这里返回的是**去掉整行头之后的正文**，交由后续清洗处理。
    """
    headers: list[str] = []
    body_lines: list[str] = []
    for line in raw.splitlines():
        if line.lstrip().startswith(">"):
            headers.append(line.lstrip()[1:].strip())
            continue
        body_lines.append(line)
    return "\n".join(body_lines), headers


def parse_fasta(text: str) -> list[FastaRecord]:
    """解析 FASTA 文本；无 ``>`` 头时按单条序列处理。"""
    records: list[FastaRecord] = []
    header: str | None = None
    buffer: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            if header is not None or buffer:
                records.append(FastaRecord(header or "unnamed", "".join(buffer)))
                buffer = []
            header = stripped[1:].strip() or "unnamed"
        else:
            buffer.append(stripped)

    if header is not None or buffer:
        records.append(FastaRecord(header or "unnamed", "".join(buffer)))

    return [record for record in records if record.sequence]


def sequence_sha256(sequence: str) -> str:
    """序列哈希，用于去重与缓存键。"""
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def validate_sequence(
    raw_sequence: str,
    *,
    strict: bool = False,
    max_length: int | None = None,
) -> SequenceCheck:
    """校验并清洗序列。

    Args:
        raw_sequence: 原始输入。**允许直接粘贴 FASTA 文本**（含 ``>`` 头行），
            也允许纯序列、带换行或空白的序列；头行会被整行忽略并给出提示。
        strict: 为 ``True`` 时，歧义残基视为错误（突变设计入口使用）。
        max_length: 长度上限，默认取配置 ``max_sequence_length``。
    """
    settings = get_settings()
    limit = max_length or settings.max_sequence_length

    # FASTA 头必须**先整行剥离**再清洗（理由见 split_fasta_headers 的文档）。
    # 放在公共校验层，结构预测 / 性质预测 / 突变设计所有入口一并受益。
    body, fasta_headers = split_fasta_headers(raw_sequence)

    cleaned, removed = clean_raw_sequence(body)
    errors: list[str] = []
    warnings: list[str] = []
    invalid_chars: dict[str, int] = {}
    ambiguous_positions: list[int] = []

    if len(fasta_headers) > 1:
        errors.append(
            f"检测到 {len(fasta_headers)} 条 FASTA 记录，本接口一次只处理 1 条序列，"
            "请仅保留目标序列后重试。"
        )
    elif fasta_headers and cleaned:
        # "可含 FASTA 头"是接口约定，忽略头属于正常输入形态：提示即可，不报错。
        preview = fasta_headers[0][:70] + ("…" if len(fasta_headers[0]) > 70 else "")
        warnings.append(f"已识别并忽略 FASTA 头行：{preview}")

    for index, char in enumerate(cleaned):
        if char in STANDARD_AA:
            continue
        if char in AMBIGUOUS_AA:
            ambiguous_positions.append(index)
            continue
        # 非氨基酸字母或其它被保留的符号
        if _CLEANUP_PATTERN.fullmatch(char) is None:
            invalid_chars[char] = invalid_chars.get(char, 0) + 1

    if not cleaned:
        if fasta_headers:
            # 只给了头没给序列（例如整段粘在一行、或头行后面是空行）。
            # 这时若沿用"序列为空"的提示，用户会反复检查序列本身而找不到原因，
            # 必须明确指出是格式问题。
            errors.append(
                "只解析到 FASTA 头，没有找到序列内容。请按标准 FASTA 格式输入："
                "头行以 > 开头独占一行，序列放在下一行。"
            )
        else:
            errors.append("序列为空：请粘贴氨基酸序列或上传 FASTA 文件。")

    if invalid_chars:
        detail = "、".join(f"{char}({count})" for char, count in sorted(invalid_chars.items()))
        errors.append(f"存在非法字符：{detail}。序列仅允许 20 种标准氨基酸。")

    if len(cleaned) > limit:
        errors.append(f"序列长度 {len(cleaned)} 超过上限 {limit}，请拆分为结构域后分别分析。")

    if len(cleaned) < 10 and cleaned:
        warnings.append(f"序列仅 {len(cleaned)} 个残基，统计类指标与结构预测可靠性较低。")

    if removed:
        detail = "、".join(
            f"{'空白' if key == 'whitespace' else key}×{count}" for key, count in removed.items()
        )
        message = f"已自动清理非残基字符：{detail}。"
        # 严格模式（突变设计入口）下，数字、'*'、'-' 这类字符几乎一定是误粘贴或格式
        # 错误（例如把序号或终止符粘进了序列）。此时**必须报错而不是警告**——
        # 静默清洗会让用户以为输入的是一条干净序列，而设计结果建立在错误的输入上。
        non_whitespace = {key: count for key, count in removed.items() if key != "whitespace"}
        if strict and non_whitespace:
            errors.append(
                message
                + " 严格模式下不接受被清理的字符（可能是误粘贴的序号或终止符），"
                "请核对原始序列后重新提交。"
            )
        else:
            warnings.append(message)

    if ambiguous_positions:
        counts = {}
        for pos in ambiguous_positions:
            char = cleaned[pos]
            counts[char] = counts.get(char, 0) + 1
        detail = "、".join(
            f"{char}×{count}（{AMBIGUOUS_MEANING.get(char, '未知')}）" for char, count in counts.items()
        )
        message = f"检测到歧义残基：{detail}，位置 {ambiguous_positions[:20]}{'…' if len(ambiguous_positions) > 20 else ''}。"
        if strict:
            errors.append(message + " 突变设计要求全部为标准氨基酸，请先人工确认。")
        else:
            warnings.append(message + " 性质预测结果置信度会相应下降。")

    return SequenceCheck(
        sequence=cleaned,
        length=len(cleaned),
        sha256=sequence_sha256(cleaned),
        invalid_chars=invalid_chars,
        ambiguous_positions=ambiguous_positions,
        warnings=warnings,
        errors=errors,
    )


def require_valid_sequence(raw_sequence: str, *, strict: bool = False) -> SequenceCheck:
    """校验失败即抛 :class:`SequenceError`，用于服务层入口。"""
    from ...core.errors import SequenceError

    check = validate_sequence(raw_sequence, strict=strict)
    if not check.ok:
        raise SequenceError("; ".join(check.errors), detail={"warnings": check.warnings})
    return check
