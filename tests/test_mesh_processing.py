import numpy as np
import pytest
import trimesh

from export_engine import export_textured_obj
from mesh_processing import load_uv_mesh
from texture_engine import DeterministicTestProvider, TextureRequest


SQUARE = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"


def write_obj(tmp_path, text):
    path = tmp_path / "source.obj"
    path.write_text(text, encoding="ascii")
    return path


def test_missing_uvs_make_padded_disjoint_charts_for_planar_quad(tmp_path):
    path = write_obj(tmp_path, SQUARE + "f -4 -3 -2 -1\n")
    mesh = load_uv_mesh(path)
    assert len(mesh.faces) == 2
    assert mesh.area == pytest.approx(1)
    charts = mesh.visual.uv[mesh.faces]
    assert ((0 < charts) & (charts < 1)).all()
    assert charts[0, :, 0].max() < charts[1, :, 0].min()
    np.testing.assert_array_equal(load_uv_mesh(path).visual.uv, mesh.visual.uv)


def test_authored_uvs_survive_seams_and_relative_indices(tmp_path):
    path = write_obj(tmp_path, SQUARE + (
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0.25 0.5\nvt 0 1\n"
        "f 1/1 2/2 3/3\nf -4/-2 -2/-3 -1/-1\n"
    ))
    mesh = load_uv_mesh(path)
    assert len(mesh.vertices) == 5
    np.testing.assert_array_equal(
        mesh.visual.uv[mesh.faces],
        [[[0, 0], [1, 0], [1, 1]], [[0.25, 0.5], [1, 1], [0, 1]]],
    )


def test_mixed_uv_references_preserve_authored_corners(tmp_path):
    path = write_obj(tmp_path, SQUARE + "vt 0.3 0.7\nf 1/1 2 3\nf 1 3 4\n")
    mesh = load_uv_mesh(path)
    np.testing.assert_array_equal(mesh.visual.uv[mesh.faces[0, 0]], [0.3, 0.7])
    assert ((0 < mesh.visual.uv) & (mesh.visual.uv < 1)).all()


@pytest.mark.parametrize(("source", "message"), [
    ("", "vertices and faces"),
    (SQUARE + "f 0 2 3\n", "index is out of bounds"),
    (SQUARE + "f 1 2 5\n", "index is out of bounds"),
    (SQUARE + "f 1 2 2\n", "degenerate"),
    (SQUARE.replace("v 0 0 0", "v nan 0 0") + "f 1 2 3\n", "finite"),
    (SQUARE + "vt 1.1 0\nf 1/1 2 3\n", "within"),
    (SQUARE + "vt 0 inf\nf 1/1 2 3\n", "finite"),
    (SQUARE + "f 1/1 2 3\n", "index is out of bounds"),
])
def test_invalid_obj_is_rejected(tmp_path, source, message):
    with pytest.raises(ValueError, match=message):
        load_uv_mesh(write_obj(tmp_path, source))


def test_export_round_trips_geometry_uvs_and_texture(tmp_path):
    path = write_obj(tmp_path, SQUARE + "f 1 2 3 4\n")
    texture = tmp_path / "input texture.png"
    data = DeterministicTestProvider().generate(TextureRequest("stone tile", path))
    texture.write_bytes(data)
    output = tmp_path / "output"
    bundle = export_textured_obj(path, texture, output)
    second = export_textured_obj(path, texture, output)

    assert bundle.obj_path.parent != second.obj_path.parent
    assert bundle.texture_path.read_bytes() == texture.read_bytes() == data
    assert f"mtllib {bundle.mtl_path.name}" in bundle.obj_path.read_text()
    assert "usemtl diffuse" in bundle.obj_path.read_text()
    assert "newmtl diffuse" in bundle.mtl_path.read_text()
    assert f"map_Kd {bundle.texture_path.name}" in bundle.mtl_path.read_text()
    scene = trimesh.load_scene(bundle.obj_path, process=False, skip_materials=True)
    loaded, = scene.geometry.values()
    expected = load_uv_mesh(path)
    np.testing.assert_array_equal(loaded.vertices[loaded.faces], expected.vertices[expected.faces])
    np.testing.assert_array_equal(loaded.visual.uv[loaded.faces], expected.visual.uv[expected.faces])
    assert path.read_text() == SQUARE + "f 1 2 3 4\n"


def test_invalid_png_creates_no_bundle(tmp_path):
    path = write_obj(tmp_path, SQUARE + "f 1 2 3\n")
    texture = tmp_path / "invalid.png"
    texture.write_bytes(b"invalid")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="not a PNG"):
        export_textured_obj(path, texture, output)
    assert not output.exists()
