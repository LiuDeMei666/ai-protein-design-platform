"""序列校验与 FASTA 上传路由。"""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import PlainTextResponse

from ...core.config import get_settings
from ...core.errors import ValidationError
from ...schemas.sequence import (
    FastaParseIn,
    FastaRecordOut,
    SequenceValidateIn,
    SequenceValidateOut,
)
from ...services.sequence.feature_utils import composition
from ...services.sequence.validator import parse_fasta, validate_sequence

router = APIRouter(prefix="/sequence", tags=["sequence"])


def _check_to_out(check) -> SequenceValidateOut:
    return SequenceValidateOut(
        ok=check.ok,
        design_ready=check.design_ready,
        sequence=check.sequence,
        length=check.length,
        sha256=check.sha256,
        invalid_chars=check.invalid_chars,
        ambiguous_positions=check.ambiguous_positions,
        ambiguous_summary=check.ambiguous_summary,
        composition={key: round(value, 5) for key, value in composition(check.sequence).items()},
        warnings=check.warnings,
        errors=check.errors,
    )


@router.post("/validate", response_model=SequenceValidateOut, summary="校验序列")
def validate(payload: SequenceValidateIn) -> SequenceValidateOut:
    """清洗并校验序列，返回非法字符位置、歧义残基与组成摘要。

    ``strict=true`` 用于突变设计前置校验（歧义残基视为错误）。
    """
    return _check_to_out(validate_sequence(payload.sequence, strict=payload.strict))


@router.post("/parse-fasta", response_model=list[FastaRecordOut], summary="解析 FASTA")
def parse(payload: FastaParseIn) -> list[FastaRecordOut]:
    """解析多序列 FASTA，返回每条记录的头部与长度。"""
    records = parse_fasta(payload.text)
    if not records:
        raise ValidationError("未解析到任何序列：请确认内容包含 FASTA 格式或以序列正文开头")
    return [
        FastaRecordOut(
            name=record.name,
            header=record.header,
            length=len("".join(record.sequence.split())),
        )
        for record in records
    ]


@router.post("/upload", response_class=PlainTextResponse, summary="上传 FASTA 文件")
async def upload(file: UploadFile = File(...)) -> PlainTextResponse:
    """上传 FASTA/文本文件，返回其文本内容（供前端填入输入框）。

    限制文件大小以避免误传大文件占满内存。
    """
    settings = get_settings()
    raw = await file.read()
    limit_bytes = int(settings.max_upload_mb * 1024 * 1024)
    if len(raw) > limit_bytes:
        raise ValidationError(
            f"文件大小 {len(raw) / 1024 / 1024:.1f} MB 超过上限 {settings.max_upload_mb} MB"
        )

    text: str | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValidationError("文件编码无法识别，请另存为 UTF-8")

    return PlainTextResponse(text)
