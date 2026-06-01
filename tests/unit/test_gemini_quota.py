from google.api_core.exceptions import ResourceExhausted

import pytest

from self_evaluating_rag.domain.exceptions.domain_errors import GeminiQuotaExceededError
from self_evaluating_rag.infrastructure.llm.gemini_client import raise_if_gemini_quota_exceeded


def test_raise_if_gemini_quota_exceeded_direct():
    with pytest.raises(GeminiQuotaExceededError) as exc_info:
        raise_if_gemini_quota_exceeded(
            ResourceExhausted("Quota exceeded"),
            model="gemini-2.5-flash",
        )
    assert exc_info.value.model == "gemini-2.5-flash"


def test_raise_if_gemini_quota_exceeded_wrapped_cause():
    try:
        raise ResourceExhausted("Quota exceeded")
    except ResourceExhausted as exc:
        with pytest.raises(GeminiQuotaExceededError):
            raise_if_gemini_quota_exceeded(exc, model="gemini-2.5-flash")


def test_raise_if_gemini_quota_exceeded_other_error_passthrough():
    exc = RuntimeError("other")
    raise_if_gemini_quota_exceeded(exc, model="gemini-2.5-flash")
    with pytest.raises(RuntimeError, match="other"):
        raise exc
