"""领域异常与统一错误响应。

分层目的：把"用户输入问题"与"外部依赖故障"区分开，前端据此给出不同引导，
避免所有失败都表现为 500。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """统一错误响应体（FastAPI exception handler 输出）。"""

    error: str = Field(description="机器可读的错误码")
    message: str = Field(description="面向用户的中文说明")
    detail: Any | None = Field(default=None, description="附加信息，如非法字符、越界位置")


class PlatformError(Exception):
    """平台异常基类。"""

    code = "platform_error"
    status_code = 500

    def __init__(self, message: str, detail: Any | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class ValidationError(PlatformError):
    """输入不合法（序列字符集、长度、坐标越界等）。"""

    code = "validation_error"
    status_code = 400


class SequenceError(ValidationError):
    """序列相关校验失败。"""

    code = "sequence_error"


class NotFoundError(PlatformError):
    """资源不存在。"""

    code = "not_found"
    status_code = 404


class ExternalServiceError(PlatformError):
    """外部依赖（ESM Atlas 等）不可用。"""

    code = "external_service_error"
    status_code = 503


class ProviderUnavailableError(ExternalServiceError):
    """结构预测 Provider 全部不可用。"""

    code = "provider_unavailable"


class ModelNotAvailableError(PlatformError):
    """模型权重缺失或加载失败。"""

    code = "model_not_available"
    status_code = 503


class JobError(PlatformError):
    """作业执行失败。"""

    code = "job_error"


class AuthError(PlatformError):
    """鉴权失败。"""

    code = "auth_error"
    status_code = 401
