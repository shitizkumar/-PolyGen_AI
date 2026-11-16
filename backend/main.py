from fastapi import FastAPI, HTTPException

from backend.config import PROJECT_ROOT
from backend.schemas import (
    GenerationResponse,
    HealthResponse,
    PromptRequest,
)
from generation_engine.text_to_3d import generate_mesh


app = FastAPI(
    title="PolyGen AI",
    description="Generative Text-to-3D Asset Pipeline",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        project="PolyGen AI",
    )


@app.get("/generation/status")
def generation_status():
    return {
        "available": False,
        "model": "shap-e",
        "message": "Model inference is not configured for this environment.",
    }


@app.post("/generate", response_model=GenerationResponse)
def generate_asset(payload: PromptRequest):
    raise HTTPException(
        status_code=503,
        detail="Text-to-3D generation is not available yet.",
    )
