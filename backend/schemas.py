from pydantic import BaseModel, Field, field_validator


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=500)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        value = value.strip()

        if not 3 <= len(value) <= 500:
            raise ValueError(
                "Prompt must contain between 3 and 500 non-space characters."
            )

        return value


class GenerationResponse(BaseModel):
    status: str
    message: str
    model_path: str | None = None


class HealthResponse(BaseModel):
    status: str
    project: str
