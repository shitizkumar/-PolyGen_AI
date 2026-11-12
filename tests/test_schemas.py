import pytest
from pydantic import ValidationError

from backend.schemas import PromptRequest, GenerationResponse, HealthResponse


def test_valid_prompt_is_trimmed():
    request = PromptRequest(prompt="  A medieval sword  ")
    assert request.prompt == "A medieval sword"


def test_prompt_rejects_too_short_input():
    with pytest.raises(ValidationError):
        PromptRequest(prompt="Hi")


def test_prompt_rejects_blank_input():
    with pytest.raises(ValidationError):
        PromptRequest(prompt="   ")


def test_prompt_rejects_excessive_length():
    with pytest.raises(ValidationError):
        PromptRequest(prompt="a" * 501)


def test_generation_response_schema():
    response = GenerationResponse(
        status="pending",
        message="Generation has not started."
    )
    assert response.status == "pending"
    assert response.model_path is None


def test_health_response_schema():
    response = HealthResponse(status="ok", project="PolyGen AI")
    assert response.status == "ok"
