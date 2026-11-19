"""Load OBJ geometry and fill missing UVs with a CPU triangle atlas."""

from io import StringIO
from math import isqrt
from pathlib import Path

import numpy as np
import trimesh


def _obj_index(value: str, count: int) -> int:
    index = int(value)
    if index < 0:
        index += count + 1
    if not 1 <= index <= count:
        raise ValueError("OBJ face index is out of bounds.")
    return index


def _normalized_obj(source: str) -> str:
    # Use a sentinel for missing UVs so trimesh can retain mixed UV references
    # and unmerge seams without loading the source's external materials.
    lines = ["vt 2 2"]
    vertex_count = uv_count = normal_count = 0
    for line in source.replace("\\\n", " ").splitlines():
        fields = line.partition("#")[0].split()
        if not fields:
            continue
        kind, *values = fields
        if kind in {"v", "vt"}:
            size = 3 if kind == "v" else 2
            if len(values) < (1 if kind == "vt" else size):
                raise ValueError(f"Incomplete OBJ {kind} record.")
            coords = [float(value) for value in values[:size]]
            if not np.isfinite(coords).all():
                raise ValueError(f"OBJ {kind} coordinates must be finite.")
            if kind == "vt":
                if any(not 0 <= value <= 1 for value in coords):
                    raise ValueError("OBJ UVs must be within [0, 1].")
                uv_count += 1
                if len(coords) == 1:
                    coords.append(0.0)
            else:
                vertex_count += 1
            lines.append(f"{kind} " + " ".join(map(str, coords)))
        elif kind == "vn":
            normal_count += 1
        elif kind == "f":
            if len(values) < 3:
                raise ValueError("OBJ faces must have at least three corners.")
            corners = []
            for value in values:
                indices = value.split("/")
                if len(indices) > 3:
                    raise ValueError("Invalid OBJ face corner.")
                vertex = _obj_index(indices[0], vertex_count)
                uv = 1
                if len(indices) > 1 and indices[1]:
                    uv = _obj_index(indices[1], uv_count) + 1
                if len(indices) == 3 and indices[2]:
                    _obj_index(indices[2], normal_count)
                corners.append(f"{vertex}/{uv}")
            for index in range(1, len(corners) - 1):
                lines.append(f"f {corners[0]} {corners[index]} {corners[index + 1]}")
    return "\n".join(lines)


def _validate_geometry(mesh: trimesh.Trimesh) -> None:
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("OBJ must contain vertices and faces.")

    with np.errstate(over="ignore", invalid="ignore"):
        triangles = mesh.vertices[mesh.faces]
        edges = triangles[:, 1:] - triangles[:, :1]
        scale = np.max(np.abs(edges), axis=(1, 2))
        if not np.isfinite(scale).all() or np.any(scale == 0):
            raise ValueError("OBJ contains degenerate or invalid triangles.")
        edges /= scale[:, None, None]
        cross = np.cross(edges[:, 0], edges[:, 1])
    if np.any(~np.any(cross != 0, axis=1)):
        raise ValueError("OBJ contains degenerate triangles.")


def _fill_uvs(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    uv = np.asarray(mesh.visual.uv)
    corners = uv[mesh.faces]
    missing = corners[:, :, 0] == 2
    affected = np.flatnonzero(missing.any(axis=1))
    if not len(affected):
        return mesh

    # Each affected triangle gets a padded cell. Keep authored corner values
    # even when only part of a face needs UVs.
    grid = isqrt(len(affected) - 1) + 1
    cells = np.arange(len(affected))
    origins = np.column_stack((cells % grid, cells // grid))
    chart = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]])
    generated = (origins[:, None, :] + chart) / grid
    faces = mesh.faces.copy()
    face_ids, corner_ids = np.where(missing)
    source = faces[face_ids, corner_ids]
    faces[face_ids, corner_ids] = len(mesh.vertices) + np.arange(len(source))
    mapped = trimesh.Trimesh(
        vertices=np.vstack((mesh.vertices, mesh.vertices[source])),
        faces=faces,
        process=False,
        visual=trimesh.visual.TextureVisuals(
            uv=np.vstack((uv, generated[missing[affected]])),
            material=mesh.visual.material,
        ),
    )
    mapped.remove_unreferenced_vertices()
    return mapped


def load_uv_mesh(path: Path) -> trimesh.Trimesh:
    """Load an OBJ, preserve authored UVs and return a UV-mapped triangle mesh."""
    path = Path(path)
    if path.suffix.lower() != ".obj":
        raise ValueError("Mesh must be an OBJ file.")
    try:
        source = _normalized_obj(path.read_text(encoding="utf-8-sig"))
        scene = trimesh.load_scene(
            StringIO(source), file_type="obj", process=False, skip_materials=True
        )
        # Normalization creates one geometry. Read it directly to avoid copying
        # trimesh's default placeholder texture, which can require Pillow.
        mesh = next(iter(scene.geometry.values()), None)
    except (OSError, ValueError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError(f"Unable to load OBJ: {exc}") from exc
    _validate_geometry(mesh)
    mesh.visual.material = trimesh.visual.material.SimpleMaterial()
    mesh = _fill_uvs(mesh)
    uv = np.asarray(mesh.visual.uv)
    if uv.shape != (len(mesh.vertices), 2) or not ((0 <= uv) & (uv <= 1)).all():
        raise ValueError("Mesh must have finite UVs within [0, 1].")
    return mesh
