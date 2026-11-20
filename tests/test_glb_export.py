from io import BytesIO
import json
import struct
import zlib

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from backend import main
from export_engine import export_textured_glb
from export_engine import glb
from mesh_processing import load_uv_mesh
from texture_engine.validation import PNG_SIGNATURE


@pytest.fixture
def textured_mesh(tmp_path):
    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    mesh = tmp_path / "source.obj"
    mesh.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0.25 0.5\nvt 0 1\n"
        "f 1/1 2/2 3/3\nf 1/4 3/3 4/5\n",
        encoding="ascii",
    )
    texture = tmp_path / "source.png"
    texture.write_bytes(
        PNG_SIGNATURE
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes([0, 255, 0, 0, 255, 0, 255, 0, 128])))
        + chunk(b"IEND", b"")
    )
    return mesh, texture


def test_glb_round_trip_and_api(tmp_path, monkeypatch, textured_mesh):
    mesh, texture = textured_mesh
    output = tmp_path / "exports"
    monkeypatch.setattr(main, "GLB_EXPORT_DIR", output)
    client = TestClient(main.app)
    assert client.get("/artifacts/glb").json() == []

    exported = export_textured_glb(mesh, texture, output).glb_path
    second = export_textured_glb(mesh, texture, output).glb_path
    assert exported != second
    assert exported.parent == output.resolve()
    (output / "legacy.obj").write_bytes(b"legacy")
    (output / ("asset_" + "0" * 32 + ".glb")).mkdir()

    assert client.get("/artifacts/glb").json() == sorted([exported.name, second.name])
    url = f"/artifacts/glb/{exported.name}"
    response = client.get(url)
    assert response.status_code == 200
    assert response.headers["content-type"] == "model/gltf-binary"
    assert response.headers["content-disposition"] == f'inline; filename="{exported.name}"'
    assert response.content == exported.read_bytes()

    scene = trimesh.load_scene(
        BytesIO(response.content), file_type="glb", process=False, skip_materials=True
    )
    loaded, = scene.geometry.values()
    source = load_uv_mesh(mesh)
    np.testing.assert_array_equal(loaded.vertices[loaded.faces], source.vertices[source.faces])
    json_size, = struct.unpack_from("<I", response.content, 12)
    tree = json.loads(response.content[20:20 + json_size])
    binary = response.content[28 + json_size:]
    primitive = tree["meshes"][0]["primitives"][0]
    accessor = tree["accessors"][primitive["attributes"]["TEXCOORD_0"]]
    view = tree["bufferViews"][accessor["bufferView"]]
    uv = np.frombuffer(binary, dtype="<f4", count=accessor["count"] * 2,
                       offset=view["byteOffset"]).reshape(-1, 2)
    corners = uv[loaded.faces]
    corners[..., 1] = 1 - corners[..., 1]
    np.testing.assert_allclose(corners, source.visual.uv[source.faces])
    material = tree["materials"][primitive["material"]]
    assert material["alphaMode"] == "BLEND"
    texture_index = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    image = tree["images"][tree["textures"][texture_index]["source"]]
    view = tree["bufferViews"][image["bufferView"]]
    assert image["mimeType"] == "image/png"
    assert binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]] == texture.read_bytes()

    download = client.get(url, params={"download": "true"})
    assert download.status_code == 200
    assert download.headers["content-disposition"] == f'attachment; filename="{exported.name}"'
    assert download.content == response.content
    assert client.get("/artifacts/glb/legacy.obj").status_code == 404
    assert client.get("/artifacts/glb/%2e%2e%2fsource.png").status_code == 404
    assert client.get("/artifacts/glb/asset_" + "f" * 32 + ".glb").status_code == 404


def test_failed_glb_validation_leaves_no_files(tmp_path, monkeypatch, textured_mesh):
    output = tmp_path / "exports"

    def reject(path, *_):
        assert path.is_file()
        assert list(output.glob("*.glb")) == []
        raise ValueError("GLB failed validation")

    monkeypatch.setattr(glb, "_validate_round_trip", reject)
    with pytest.raises(ValueError, match="failed validation"):
        export_textured_glb(*textured_mesh, output)

    assert list(output.iterdir()) == []
