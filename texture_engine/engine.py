"""Generate a standalone PNG associated with an existing mesh.

This stage does not unwrap UVs, bake textures, or modify mesh materials.
Providers are explicitly selected; there is no default synthetic fallback.
"""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from backend.config import ARTIFACTS_DIR
from generation_engine.text_to_3d import normalize_prompt, validate_mesh_file
from texture_engine.providers import TextureProvider, TextureRequest, TextureResult
from texture_engine.validation import TextureError, TextureValidationError, validate_png


class TextureEngine:
    def __init__(
        self, provider: TextureProvider, output_dir: Path = ARTIFACTS_DIR / "textures"
    ):
        self.provider = provider
        # This directory is trusted application configuration, never prompt input.
        self.output_dir = Path(output_dir).resolve()

    def generate(
        self, prompt: str, mesh_path: Path, *, depth_map_path: Path | None = None
    ) -> TextureResult:
        prompt = normalize_prompt(prompt)
        if depth_map_path is not None and not self.provider.supports_depth_conditioning:
            raise TextureValidationError("Depth-conditioned ControlNet is not implemented.")
        try:
            mesh_path = Path(mesh_path).resolve(strict=True)
            validate_mesh_file(mesh_path)
        except (OSError, ValueError) as exc:
            raise TextureValidationError("An existing, valid mesh is required.") from exc
        request = TextureRequest(prompt, mesh_path, depth_map_path)
        try:
            data = self.provider.generate(request)
        except Exception as exc:
            raise TextureError(f"Texture provider '{self.provider.name}' failed.") from exc
        width, height = validate_png(data)
        # A fixed prefix and UUID prevent prompt traversal, reserved names and
        # collisions. Linking the complete file refuses existing destinations.
        prefix = "test_texture" if self.provider.is_test_provider else "texture"
        path = self.output_dir / f"{prefix}_{uuid4().hex}.png"
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output = NamedTemporaryFile(dir=self.output_dir, suffix=".tmp", delete=False)
            temporary_path = Path(output.name)
            try:
                with output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                os.link(temporary_path, path)
            finally:
                temporary_path.unlink()
        except OSError as exc:
            raise TextureError("Unable to save the texture PNG.") from exc
        return TextureResult(
            path, mesh_path, self.provider.name, self.provider.is_test_provider,
            width, height,
        )
