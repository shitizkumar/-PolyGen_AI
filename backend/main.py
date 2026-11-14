from fastapi import FastAPI, HTTPException

from backend.config import PROJECT_ROOT
from backend.schemas import (
    PromptRequest,
    GenerationResponse,
    HealthResponse,
)
from generation_engine.text_to_3d import generate_mesh


app = FastAPI(
    title="PolyGen AI",
    description="Generative Text-to-3D Asset Pipeline",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check backend availability."""
    return HealthResponse(
        status="ok",
        project="PolyGen AI",
    )


@app.post("/generate", response_model=GenerationResponse)
def generate_asset(payload: PromptRequest):
    """Generate a 3D asset from a text prompt."""
    try:
        model_path = generate_mesh(payload.prompt)

        relative_path = model_path.relative_to(PROJECT_ROOT)

        return GenerationResponse(
            status="completed",
            message="3D asset generated and validated successfully.",
            model_path=relative_path.as_posix(),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"3D generation failed: {exc}",
        ) from exc
