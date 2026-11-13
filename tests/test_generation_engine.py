import pytest

from generation_engine.text_to_3d import (
    normalize_prompt,
    build_output_path,
)


def test_prompt_is_normalized():
    assert normalize_prompt("  A medieval sword  ") == "A medieval sword"


def test_short_prompt_is_rejected():
    with pytest.raises(ValueError):
        normalize_prompt("Hi")


def test_blank_prompt_is_rejected():
    with pytest.raises(ValueError):
        normalize_prompt("   ")


def test_long_prompt_is_rejected():
    with pytest.raises(ValueError):
        normalize_prompt("a" * 501)


def test_output_path_uses_obj_extension(tmp_path):
    path = build_output_path("A medieval sword", tmp_path)

    assert path.suffix == ".obj"
    assert path.parent == tmp_path


def test_output_filename_is_safe(tmp_path):
    path = build_output_path(
        "A sword / with special characters!",
        tmp_path,
    )

    assert "/" not in path.name
    assert "\\" not in path.name

import trimesh

from generation_engine.text_to_3d import validate_mesh_file


def test_valid_obj_mesh_passes_validation(tmp_path):
    mesh_path = tmp_path / "valid_mesh.obj"
    trimesh.creation.box().export(mesh_path)

    result = validate_mesh_file(mesh_path)

    assert result["vertex_count"] > 0
    assert result["face_count"] > 0


def test_invalid_obj_mesh_is_rejected(tmp_path):
    mesh_path = tmp_path / "invalid_mesh.obj"
    mesh_path.write_text(
        "This is not a valid mesh",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        validate_mesh_file(mesh_path)
