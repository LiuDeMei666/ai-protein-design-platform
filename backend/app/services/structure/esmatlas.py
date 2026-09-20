"""在线 ESM Atlas 结构预测 Provider（主通道）。

接口
----
``POST https://api.esmatlas.com/foldSequence/v1/pdb/``
请求体为**纯序列字符串**（``Content-Type: application/x-www-form-urlencoded``），
响应体为 PDB 文本。

实测（本机）：72 aa 序列 -> HTTP 200，2.26s，42486 字节，506 个 ATOM 记录，
HEADER 标注 ``ESMFOLD V1 PREDICTION FOR INPUT``，许可为 CC-BY-4.0（可商用）。

工程处理
--------
* 并发用信号量限制（默认 2），避免被服务端限流。
* 超时 + 指数退避重试（tenacity）。
* 超过 ``esmatlas_max_length`` 的序列交给 :mod:`domain_splitter` 分片预测，
  分片边界与"接口未建模"的事实全部回传到 ``warnings``。
"""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ...core.config import get_settings
from ...core.errors import ExternalServiceError, ValidationError
from ...core.logging import describe_sequence, get_logger
from . import pdb_utils
from .base import BasePredictor, StructureResult
from .domain_splitter import merge_fragment_pdbs, plan_split

logger = get_logger(__name__)

#: 单条序列最多允许切分成多少片段，超过则判定为"不适合在线折叠"
MAX_FRAGMENTS = 12


class ESMAtlasPredictor(BasePredictor):
    """ESM Atlas 在线折叠。"""

    name = "esmatlas"
    model_version = "ESMFold v1 (ESM Metagenomic Atlas, CC-BY-4.0)"

    def __init__(self) -> None:
        settings = get_settings()
        self.endpoint = settings.esmatlas_url
        self.max_length = settings.esmatlas_max_length
        self.timeout = settings.esmatlas_timeout
        self.retries = max(1, settings.esmatlas_retries)
        self._semaphore = threading.Semaphore(max(1, settings.esmatlas_concurrency))
        self._reachable: bool | None = None
        self._reachable_checked_at: float = 0.0
        self._reachable_ttl = 60.0
        self._now: float = 0.0

    # ------------------------------------------------------------------ #
    # 可用性
    # ------------------------------------------------------------------ #
    def available(self) -> bool:
        """轻量连通性探测，结果缓存 60 秒，避免健康检查拖慢。"""
        now = time.monotonic()
        if self._reachable is not None and (now - self._reachable_checked_at) < self._reachable_ttl:
            return self._reachable

        reachable = False
        try:
            with httpx.Client(timeout=4.0, follow_redirects=True) as client:
                # 任意 HTTP 响应都说明网络可达（GET 该接口预期返回 4xx）
                client.get(self.endpoint)
                reachable = True
        except Exception as exc:
            logger.warning("ESM Atlas 连通性探测失败: %s", exc)

        self._reachable = reachable
        self._reachable_checked_at = now
        return reachable

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available(),
            "max_length": self.max_length,
            "model_version": self.model_version,
            "endpoint": self.endpoint,
            "timeout_seconds": self.timeout,
            "concurrency": get_settings().esmatlas_concurrency,
            "license": "CC-BY-4.0（学术与商业用途均可）",
            "note": "在线接口，企业内网离线环境下不可用；超长序列自动分片",
        }

    # ------------------------------------------------------------------ #
    # 请求
    # ------------------------------------------------------------------ #
    def _post_once(self, sequence: str) -> str:
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            response = client.post(
                self.endpoint,
                content=sequence.encode("ascii"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if response.status_code >= 500:
            raise httpx.HTTPStatusError(
                f"服务端错误 {response.status_code}",
                request=response.request,
                response=response,
            )
        if response.status_code != 200:
            snippet = response.text[:200].replace("\n", " ")
            raise ExternalServiceError(
                f"ESM Atlas 返回异常状态 {response.status_code}",
                detail={"body": snippet},
            )
        body = response.text
        if "ATOM" not in body and "CA" not in body:
            snippet = body[:200].replace("\n", " ")
            raise ExternalServiceError(
                "ESM Atlas 未返回有效 PDB",
                detail={
                    "body": snippet,
                    "hint": "常见原因：序列过长或含非标准残基",
                },
            )
        return body

    def _request(self, sequence: str) -> str:
        @retry(
            stop=stop_after_attempt(max(2, 3)),
            wait=wait_exponential(multiplier=1.5, min=2, max=20),
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
            reraise=True,
        )
        def _call() -> str:
            return self._post_once(sequence)

        try:
            with self._semaphore:
                return _call()
        except RetryError as exc:  # pragma: no cover - tenacity reraise=True 时不会走到
            raise ExternalServiceError(f"ESM Atlas 重试全部失败: {exc}") from exc
        except httpx.TransportError as exc:
            raise ExternalServiceError(
                "无法连接 ESM Atlas（网络不可达或超时）",
                detail={"endpoint": self.endpoint, "error": str(exc)},
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ExternalServiceError(
                f"ESM Atlas 服务端错误: {exc}",
                detail={"endpoint": self.endpoint},
            ) from exc

    # ------------------------------------------------------------------ #
    # 预测
    # ------------------------------------------------------------------ #
    def predict(self, sequence: str, *, allow_split: bool = True) -> StructureResult:
        sequence = self._require_sequence(sequence)
        length = len(sequence)

        if length <= int(self.max_length or length):
            return self._predict_single(sequence)

        if not allow_split:
            raise ValidationError(
                f"序列长度 {length} 超过在线折叠上限 {self.max_length}，且已禁用自动分片",
                detail={"max_length": self.max_length},
            )

        return self._predict_split(sequence)

    def _predict_single(self, sequence: str) -> StructureResult:
        logger.info("ESM Atlas 折叠 %s", describe_sequence(sequence))
        started = time.perf_counter()
        pdb_text = self._request(sequence)
        elapsed = time.perf_counter() - started

        data = pdb_utils.parse_pdb(pdb_text, source=self.name)
        if data.length == 0:
            raise ExternalServiceError("ESM Atlas 返回的 PDB 无法解析出残基")

        warnings: list[str] = []
        if data.length != len(sequence):
            warnings.append(
                f"返回结构残基数 {data.length} 与输入序列长度 {len(sequence)} 不一致，"
                "可能存在缺失残基，请核对。"
            )

        plddt = pdb_utils.extract_plddt(data)
        stats = pdb_utils.build_structure_stats(data, plddt)
        stats["elapsed_seconds"] = round(elapsed, 2)

        logger.info(
            "ESM Atlas 完成 %s 残基=%d mean_plddt=%.1f 耗时=%.2fs",
            describe_sequence(sequence),
            data.length,
            sum(plddt) / len(plddt) if plddt else 0.0,
            elapsed,
        )

        return StructureResult(
            pdb_text=pdb_text,
            plddt=plddt,
            mean_plddt=round(float(sum(plddt) / len(plddt)), 2) if plddt else 0.0,
            source=self.name,
            truncated=False,
            segments=[(0, data.length)],
            model_version=self.model_version,
            stats=stats,
            warnings=warnings,
        )

    def _predict_split(self, sequence: str) -> StructureResult:
        """超长序列分片折叠后合并。"""
        plan = plan_split(sequence, int(self.max_length))
        if plan.count > MAX_FRAGMENTS:
            raise ValidationError(
                f"序列长度 {len(sequence)} 需切分为 {plan.count} 个片段，超过上限 {MAX_FRAGMENTS}。"
                "建议按结构域人工拆分后分别分析。",
                detail={"segments": plan.segments, "max_fragments": MAX_FRAGMENTS},
            )

        logger.info(
            "序列超长（%d aa），按 %d 个片段折叠，策略=%s",
            len(sequence),
            plan.count,
            plan.strategy,
        )

        pdb_texts: list[str] = []
        plddt_all: list[float] = []
        elapsed_total = 0.0
        for index, (start, end) in enumerate(plan.segments, start=1):
            fragment = sequence[start:end]
            logger.info("  片段 %d/%d: %d-%d (%d aa)", index, plan.count, start, end, len(fragment))
            started = time.perf_counter()
            pdb_text = self._request(fragment)
            elapsed_total += time.perf_counter() - started
            pdb_texts.append(pdb_text)

            fragment_data = pdb_utils.parse_pdb(pdb_text, source=self.name)
            fragment_plddt = pdb_utils.extract_plddt(fragment_data)
            if len(fragment_plddt) != len(fragment):
                # 长度不齐时按缺失补 0，保证逐残基数组与全长严格对齐
                fragment_plddt = (fragment_plddt + [0.0] * len(fragment))[: len(fragment)]
            plddt_all.extend(fragment_plddt)

        merged_pdb = merge_fragment_pdbs(pdb_texts, plan.segments)
        merged_data = pdb_utils.parse_pdb(merged_pdb, source=self.name)
        stats = pdb_utils.build_structure_stats(merged_data, plddt_all)
        stats["elapsed_seconds"] = round(elapsed_total, 2)
        stats["split_strategy"] = plan.strategy
        stats["fragment_count"] = plan.count

        warnings = [
            f"序列长度 {len(sequence)} 超过在线折叠上限 {self.max_length}，"
            f"已按 {plan.count} 个片段分别折叠后拼接。",
            "**片段之间的相对空间取向未经建模**，跨片段的接触与整体构象不可信；"
            "片段内部的二级结构与置信度评估仍然有效。",
        ]
        if plan.broken_helices:
            warnings.append(
                f"有 {plan.broken_helices} 处切点可能落在三股螺旋内部，"
                "该处结构连续性需人工复核。"
            )

        return StructureResult(
            pdb_text=merged_pdb,
            plddt=[round(value, 2) for value in plddt_all],
            mean_plddt=round(float(sum(plddt_all) / len(plddt_all)), 2) if plddt_all else 0.0,
            source=self.name,
            truncated=False,
            segments=plan.segments,
            model_version=self.model_version,
            stats=stats,
            warnings=warnings,
        )
