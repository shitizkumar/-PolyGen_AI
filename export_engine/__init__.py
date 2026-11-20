"""Export a UV-mapped OBJ and its diffuse texture as a portable bundle."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from backend.config import ARTIFACTS_DIR
from export_engine.glb import TexturedGlb, export_textured_glb
from mesh_processing import load_uv_mesh
from texture_engine.validation import MAX_PNG_BYTES, validate_png


@dataclass(frozen=True)
class TexturedObj:
    obj_path: Path
    mtl_path: Path
    texture_path: Path


def export_textured_obj(
    mesh_path: Path, texture_path: Path, output_dir: Path = ARTIFACTS_DIR / "exports"
) -> TexturedObj:
    """Process an OBJ and publish all three files in a new bundle directory."""
    mesh = load_uv_mesh(mesh_path)
    with Path(texture_path).open("rb") as texture_file:
        texture = texture_file.read(MAX_PNG_BYTES + 1)
    validate_png(texture)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"asset_{uuid4().hex}"
    with TemporaryDirectory(dir=output_dir) as staging:
        # Publish the complete directory with one rename on the same filesystem.
        bundle = Path(staging) / destination.name
        bundle.mkdir()
        with (bundle / "mesh.obj").open("w", encoding="ascii", newline="\n") as obj:
            obj.write("mtllib mesh.mtl\n")
            for x, y, z in mesh.vertices:
                obj.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
            for u, v in mesh.visual.uv:
                obj.write(f"vt {u:.17g} {v:.17g}\n")
            obj.write("usemtl diffuse\n")
            for face in mesh.faces + 1:
                obj.write("f " + " ".join(f"{index}/{index}" for index in face) + "\n")
        (bundle / "mesh.mtl").write_text(
            "newmtl diffuse\nKd 1.0 1.0 1.0\nmap_Kd texture.png\n",
            encoding="ascii",
        )
        (bundle / "texture.png").write_bytes(texture)
        bundle.rename(destination)
    return TexturedObj(destination / "mesh.obj", destination / "mesh.mtl", destination / "texture.png")
