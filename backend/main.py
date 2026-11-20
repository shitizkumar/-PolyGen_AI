from fastapi import FastAPI, HTTPException

from pathlib import Path

from fastapi.responses import FileResponse

from backend.schemas import (
    GenerationResponse,
    HealthResponse,
    PromptRequest,
)
from export_engine.glb import GLB_EXPORT_DIR, GLB_FILENAME


app = FastAPI(
    title="PolyGen AI",
    description="Generative Text-to-3D Asset Pipeline",
    version="1.0.0",
)


def _glb_artifact(filename: str) -> Path | None:
    if not GLB_FILENAME.fullmatch(filename):
        return None
    root = GLB_EXPORT_DIR.resolve()
    path = root / filename
    if path.is_symlink() or path.resolve().parent != root or not path.is_file():
        return None
    return path


@app.get("/artifacts/glb", response_model=list[str])
def list_glb_artifacts():
    if not GLB_EXPORT_DIR.is_dir():
        return []
    return sorted(
        path.name for path in GLB_EXPORT_DIR.iterdir() if _glb_artifact(path.name)
    )


@app.get("/artifacts/glb/{filename}", response_class=FileResponse)
def serve_glb_artifact(filename: str, download: bool = False):
    path = _glb_artifact(filename)
    if path is None:
        raise HTTPException(status_code=404, detail="GLB artifact not found.")
    return FileResponse(
        path,
        media_type="model/gltf-binary",
        filename=filename,
        content_disposition_type="attachment" if download else "inline",
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
