"""Export a UV-mapped OBJ and a PNG as a self-contained GLB."""

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import trimesh
from trimesh.exchange.gltf import export_glb
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from backend.config import ARTIFACTS_DIR
from mesh_processing import load_uv_mesh
from texture_engine.validation import MAX_PNG_BYTES, validate_png

GLB_EXPORT_DIR = ARTIFACTS_DIR / "exports"
GLB_FILENAME = re.compile(r"asset_[0-9a-f]{32}\.glb")


@dataclass(frozen=True)
class TexturedGlb:
    glb_path: Path


def _embed_png(buffers, tree, png: bytes) -> None:
    index = len(buffers)
    # Buffer views start on four-byte boundaries. The image view itself
    # excludes any padding, so consumers receive the original PNG bytes.
    buffers["diffuse_png"] = png + b"\0" * (-len(png) % 4)
    tree["images"] = [{"bufferView": index, "mimeType": "image/png"}]
    tree["textures"] = [{"source": 0}]
    tree["materials"][0].setdefault("pbrMetallicRoughness", {})["baseColorTexture"] = {
        "index": 0
    }


def _validate_round_trip(path: Path, mesh: trimesh.Trimesh, png: bytes) -> None:
    try:
        scene = trimesh.load_scene(path, process=False, skip_materials=True)
    except Exception as exc:
        raise ValueError("Unable to reload the exported GLB.") from exc

    if len(scene.geometry) != 1:
        raise ValueError("GLB must contain one mesh.")
    loaded = next(iter(scene.geometry.values()))
    if not (
        isinstance(loaded, trimesh.Trimesh)
        and np.array_equal(loaded.faces, mesh.faces)
        and np.isfinite(loaded.vertices).all()
        and np.array_equal(loaded.vertices, mesh.vertices.astype(np.float32))
    ):
        raise ValueError("GLB geometry changed during export.")
    if np.any(loaded.area_faces <= 0):
        raise ValueError("GLB contains degenerate triangles after float32 conversion.")

    # trimesh requires Pillow to load image materials. Verify the serialized
    # UV accessor and the material's embedded PNG directly from the GLB.
    with path.open("rb") as stream:
        stream.seek(12)
        json_size, = struct.unpack("<I", stream.read(4))
        stream.read(4)  # JSON chunk type
        tree = json.loads(stream.read(json_size))
        stream.read(8)  # BIN chunk header
        binary_start = stream.tell()

        primitive = tree["meshes"][0]["primitives"][0]
        accessor = tree["accessors"][primitive["attributes"]["TEXCOORD_0"]]
        if (accessor["componentType"], accessor["type"], accessor["count"]) != (
            5126, "VEC2", len(mesh.vertices)
        ):
            raise ValueError("GLB has an invalid UV accessor.")
        view = tree["bufferViews"][accessor["bufferView"]]
        stream.seek(binary_start + view["byteOffset"] + accessor.get("byteOffset", 0))
        uv = np.frombuffer(stream.read(accessor["count"] * 8), dtype="<f4").reshape(-1, 2)
        expected_uv = mesh.visual.uv.copy()
        expected_uv[:, 1] = 1.0 - expected_uv[:, 1]
        if not np.array_equal(uv, expected_uv.astype(np.float32)):
            raise ValueError("GLB UVs changed during export.")

        material = tree["materials"][primitive["material"]]
        pbr = material["pbrMetallicRoughness"]
        texture = tree["textures"][pbr["baseColorTexture"]["index"]]
        image = tree["images"][texture["source"]]
        view = tree["bufferViews"][image["bufferView"]]
        stream.seek(binary_start + view["byteOffset"])
        embedded = stream.read(view["byteLength"])
    if (
        image["mimeType"] != "image/png"
        or material["alphaMode"] != mesh.visual.material.alphaMode
        or pbr["metallicFactor"] != 0.0
        or pbr["roughnessFactor"] != 1.0
        or embedded != png
    ):
        raise ValueError("GLB texture material changed during export.")


def export_textured_glb(
    mesh_path: Path, texture_path: Path, output_dir: Path = GLB_EXPORT_DIR
) -> TexturedGlb:
    """Validate a textured GLB and publish it with a same-filesystem rename."""
    mesh = load_uv_mesh(mesh_path)
    with Path(texture_path).open("rb") as texture_file:
        png = texture_file.read(MAX_PNG_BYTES + 1)
    validate_png(png)
    mesh.visual = TextureVisuals(
        uv=mesh.visual.uv,
        material=PBRMaterial(
            name="diffuse",
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="BLEND" if png[25] == 6 else "OPAQUE",
        ),
    )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"asset_{uuid4().hex}.glb"
    with TemporaryDirectory(dir=output_dir) as staging:
        staged = Path(staging) / destination.name
        def embed(buffers, tree):
            _embed_png(buffers, tree, png)

        def trim_image_view(tree):
            image = tree["images"][0]
            tree["bufferViews"][image["bufferView"]]["byteLength"] = len(png)

        staged.write_bytes(export_glb(
            mesh.scene(),
            include_normals=True,
            buffer_postprocessor=embed,
            tree_postprocessor=trim_image_view,
        ))
        _validate_round_trip(staged, mesh, png)
        staged.replace(destination)
    return TexturedGlb(destination)
