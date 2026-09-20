"""长序列域切分。

背景
----
ESM Atlas 在线接口的实测单次折叠长度上限约 400 残基，而重组胶原蛋白全长常达
1000+ 残基。**绝不能静默截断**——那会让用户拿到一个"看起来完整"的残缺结构。

策略
----
按"最可能断开"的位置切割，优先级从高到低：

1. **柔性连接区**：Gly/Ser/Pro/Thr/Gln 富集的 12 残基窗口（天然铰链区）。
2. **低复杂度边界**：单一残基占比 > 50% 的区域端点。
3. **胶原三股螺旋中断点**：Gly-X-Y 周期性被破坏的位置（避免把三股螺旋从中间劈开）。
4. 兜底：固定长度切分。

所有切点都在结果中回传（:attr:`StructureResult.segments`），前端会画出片段边界。
"""

from __future__ import annotations

from dataclasses import dataclass

FLEXIBLE_RESIDUES = frozenset("GSPTQ")
WINDOW = 12


@dataclass
class SplitPlan:
    """切分方案。"""

    segments: list[tuple[int, int]]
    strategy: str
    #: 被强制切断的三股螺旋数量（0 表示切分未破坏螺旋）
    broken_helices: int = 0

    @property
    def count(self) -> int:
        return len(self.segments)

    def covers(self, length: int) -> bool:
        """校验片段是否无缝、无重叠地覆盖整条序列。"""
        if not self.segments:
            return length == 0
        if self.segments[0][0] != 0 or self.segments[-1][1] != length:
            return False
        for index in range(len(self.segments) - 1):
            if self.segments[index][1] != self.segments[index + 1][0]:
                return False
        return True


def _flexibility(sequence: str, index: int, window: int = WINDOW) -> float:
    """以 ``index`` 为中心窗口的柔性残基占比。"""
    start = max(0, index - window // 2)
    end = min(len(sequence), start + window)
    if end <= start:
        return 0.0
    segment = sequence[start:end]
    return sum(1 for char in segment if char in FLEXIBLE_RESIDUES) / len(segment)


def _low_complexity(sequence: str, index: int, window: int = 20) -> bool:
    """``index`` 是否位于低复杂度（单一残基主导）区域。"""
    start = max(0, index - window // 2)
    end = min(len(sequence), start + window)
    if end - start < 5:
        return False
    segment = sequence[start:end]
    counts: dict[str, int] = {}
    for char in segment:
        counts[char] = counts.get(char, 0) + 1
    return max(counts.values()) / len(segment) > 0.5


def _is_helix_boundary(sequence: str, index: int) -> bool:
    """判断 ``index`` 是否位于胶原三股螺旋的 Gly-X-Y 周期中断处。

    在 9 残基窗口内检查 Gly 是否保持 ``% 3`` 相位；相位被破坏即为安全切点。
    """
    start = max(0, index - 4)
    end = min(len(sequence), index + 5)
    segment = sequence[start:end]
    if segment.count("G") < 2:
        return False

    gly_positions = [i for i, char in enumerate(segment) if char == "G"]
    gaps = [
        gly_positions[i + 1] - gly_positions[i] for i in range(len(gly_positions) - 1)
    ]
    if not gaps:
        return False
    in_phase = sum(1 for gap in gaps if gap % 3 == 0)
    # 相位一致性低 -> 螺旋已中断，是安全切点
    return (in_phase / len(gaps)) < 0.6


def plan_split(
    sequence: str,
    max_length: int,
    *,
    min_segment: int = 40,
    search_from: float = 0.55,
) -> SplitPlan:
    """规划切分点。

    Args:
        sequence: 待切分序列。
        max_length: 单个片段的长度上限。
        min_segment: 最小片段长度，避免产生过短的碎片。
        search_from: 在 ``[start + max_length*search_from, end]`` 内搜索切点，
            保证片段不会过短。
    """
    length = len(sequence)
    if length <= max_length:
        return SplitPlan(segments=[(0, length)], strategy="无需切分")

    segments: list[tuple[int, int]] = []
    broken_helices = 0
    start = 0
    strategy_notes: set[str] = set()

    while start < length:
        end = min(start + max_length, length)
        if end >= length:
            segments.append((start, length))
            break

        search_lo = min(start + max(min_segment, int(max_length * search_from)), end - 10)
        if search_lo >= end - 5:
            search_lo = max(start + min_segment, end - 10)

        best_cut: int | None = None
        best_score = -1.0
        best_kind = ""

        for cut in range(search_lo, end - 4):
            if _low_complexity(sequence, cut):
                score, kind = 1.0 + _flexibility(sequence, cut), "低复杂度边界"
            elif _is_helix_boundary(sequence, cut):
                score, kind = 0.9 + _flexibility(sequence, cut), "三股螺旋中断点"
            else:
                score, kind = _flexibility(sequence, cut), "柔性连接区"

            if score > best_score:
                best_score, best_cut, best_kind = score, cut, kind

        if best_cut is None:
            best_cut, best_kind = end, "固定长度"
        else:
            if best_kind != "三股螺旋中断点" and not _is_helix_boundary(sequence, best_cut):
                # 在螺旋内部切断，记录代价
                window = sequence[max(0, best_cut - 6) : best_cut + 6]
                if window.count("G") >= 3:
                    broken_helices += 1

        segments.append((start, best_cut))
        strategy_notes.add(best_kind)
        start = best_cut

    plan = SplitPlan(
        segments=segments,
        strategy="；".join(sorted(strategy_notes)) or "固定长度",
        broken_helices=broken_helices,
    )
    if not plan.covers(length):  # pragma: no cover - 防御性校验
        plan.segments = [
            (index * max_length, min((index + 1) * max_length, length))
            for index in range((length + max_length - 1) // max_length)
        ]
        plan.strategy = "固定长度（兜底）"
    return plan


def merge_fragment_pdbs(pdb_texts: list[str], segments: list[tuple[int, int]]) -> str:
    """把多个片段的 PDB 合并为单条链，并重编号为全长坐标。

    合并后的链编号保持 ``A``，残基号按 ``segments`` 偏移还原为全长序号，
    这样前端按序列位置高亮突变热点时无需再做映射。
    """
    if not pdb_texts:
        return ""
    if len(pdb_texts) == 1:
        return pdb_texts[0]

    lines_out: list[str] = []
    serial = 1
    for pdb_text, (start, _end) in zip(pdb_texts, segments):
        for raw_line in pdb_text.splitlines():
            record = raw_line[:6].strip()
            if record not in ("ATOM", "HETATM", "TER"):
                continue
            if record == "TER":
                continue
            try:
                residue_number = int(raw_line[22:26])
            except ValueError:
                continue
            new_number = start + residue_number
            new_line = (
                f"{raw_line[:6]}{serial:5d}{raw_line[11:22]}{new_number:4d}{raw_line[26:]}"
            )
            lines_out.append(new_line.rstrip())
            serial += 1
        lines_out.append("TER")

    lines_out.append("END")
    return "\n".join(lines_out) + "\n"
