import re
from pathlib import Path
from uuid import uuid4

import numpy as np
import trimesh

from backend.config import ARTIFACTS_DIR, MAX_PROMPT_LENGTH
from generation_engine.model_loader import load_shap_e_models


def normalize_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise ValueError("Prompt must be a string.")

    prompt = prompt.strip()

    if not 3 <= len(prompt) <= MAX_PROMPT_LENGTH:
        raise ValueError(
            f"Prompt must contain between 3 and {MAX_PROMPT_LENGTH} characters."
        )

    return prompt


def build_output_path(prompt: str, output_dir: Path = ARTIFACTS_DIR) -> Path:
    """Create a unique, filesystem-safe OBJ output path."""
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    slug = slug[:40] or "generated_asset"
    filename = f"{slug}_{uuid4().hex[:8]}.obj"
    return Path(output_dir) / filename


def validate_mesh_file(mesh_path: Path) -> dict[str, int]:
    """Validate that an exported file contains usable mesh geometry."""
    mesh_path = Path(mesh_path)

    if not mesh_path.is_file() or mesh_path.stat().st_size == 0:
        raise ValueError("Mesh file is missing or empty.")

    try:
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
    except Exception as exc:
        raise ValueError("Unable to load the exported mesh.") from exc

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Mesh must contain vertices and faces.")

    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Mesh contains invalid vertex coordinates.")

    if not np.isfinite(mesh.bounds).all():
        raise ValueError("Mesh contains invalid bounds.")

    if (mesh.extents <= 0).any():
        raise ValueError("Mesh must have non-zero dimensions on all axes.")

    return {
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
    }


def generate_mesh(prompt: str, output_dir: Path = ARTIFACTS_DIR) -> Path:
    """Generate and validate a 3D mesh from text using OpenAI Shap-E."""
    prompt = normalize_prompt(prompt)

    models = load_shap_e_models()

    from shap_e.diffusion.sample import sample_latents
    from shap_e.util.notebooks import decode_latent_mesh

    latents = sample_latents(
        batch_size=1,
        model=models.text_model,
        diffusion=models.diffusion,
        guidance_scale=15.0,
        model_kwargs={"texts": [prompt]},
        progress=True,
        clip_denoised=True,
        use_fp16=models.device.type == "cuda",
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    if not latents:
        raise RuntimeError("Shap-E did not return a generated latent.")

    mesh = decode_latent_mesh(models.transmitter, latents[0]).tri_mesh()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(prompt, output_dir)

    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            mesh.write_obj(output_file)

        validate_mesh_file(output_path)

    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    return output_path
