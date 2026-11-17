"""Day 1 texture tests: temporary meshes and mocked remote inference only."""

from dataclasses import replace
from unittest.mock import Mock
import struct
import zlib

import pytest
import trimesh

from texture_engine import (
    DeterministicTestProvider,
    RemoteStableDiffusionProvider,
    TextureEngine,
    TextureError,
    TextureRequest,
    TextureValidationError,
)
from texture_engine.validation import PNG_SIGNATURE, validate_png


@pytest.fixture
def mesh_path(tmp_path):
    path = tmp_path / "mesh.obj"
    trimesh.creation.box().export(path)
    return path


@pytest.fixture
def png(mesh_path):
    return DeterministicTestProvider().generate(TextureRequest("red stone", mesh_path))


def chunk(kind, payload):
    return (
        struct.pack(">I", len(payload)) + kind + payload
        + struct.pack(">I", zlib.crc32(kind + payload))
    )


def make_png(width=1, height=1, color=2, pixels=b"\x00\x01\x02\x03"):
    return (
        PNG_SIGNATURE
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


def test_local_provider_is_deterministic_and_explicit(mesh_path, tmp_path):
    provider = DeterministicTestProvider()
    request = TextureRequest("red stone", mesh_path)
    assert provider.generate(request) == provider.generate(request)
    assert provider.generate(request) != provider.generate(replace(request, prompt="blue stone"))
    original = mesh_path.read_bytes()
    engine = TextureEngine(provider, tmp_path / "textures")
    first = engine.generate("  red stone  ", mesh_path)
    second = engine.generate("red stone", mesh_path)
    assert first.texture_path != second.texture_path
    assert first.texture_path.read_bytes() == second.texture_path.read_bytes()
    assert first.is_test_texture is True
    assert first.provider == "deterministic-test"
    assert first.texture_path.name.startswith("test_texture_")
    assert (first.width, first.height) == (8, 8)
    assert first.mesh_path == mesh_path.resolve()
    assert mesh_path.read_bytes() == original


def test_remote_provider_receives_normalized_request(mesh_path, tmp_path, png):
    transport = Mock(return_value=png)
    result = TextureEngine(
        RemoteStableDiffusionProvider(transport), tmp_path / "textures"
    ).generate("  red stone  ", mesh_path)
    transport.assert_called_once_with(TextureRequest("red stone", mesh_path.resolve()))
    assert result.is_test_texture is False
    assert result.provider == "remote-stable-diffusion"
    assert result.texture_path.read_bytes() == png


@pytest.mark.parametrize("prompt", ["", "  ", "ab", "a" * 501, None])
def test_bad_prompt_never_calls_provider(prompt, mesh_path, tmp_path):
    transport = Mock()
    with pytest.raises(ValueError):
        TextureEngine(RemoteStableDiffusionProvider(transport), tmp_path / "out").generate(
            prompt, mesh_path
        )
    transport.assert_not_called()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("contents", [None, "", "not a mesh"])
def test_invalid_mesh_never_calls_provider(contents, tmp_path):
    path = tmp_path / "bad.obj"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    transport = Mock()
    with pytest.raises(TextureValidationError, match="valid mesh"):
        TextureEngine(RemoteStableDiffusionProvider(transport), tmp_path / "out").generate(
            "red stone", path
        )
    transport.assert_not_called()


def test_output_path_ignores_prompt_paths(mesh_path, tmp_path):
    output = tmp_path / "out"
    result = TextureEngine(DeterministicTestProvider(), output).generate(
        "../../CON / back\\slash : texture", mesh_path
    )
    assert result.texture_path.parent == output.resolve()
    assert result.texture_path.suffix == ".png"


def test_remote_failure_is_chained_without_fallback(mesh_path, tmp_path):
    error = TimeoutError("remote timeout")
    transport = Mock(side_effect=error)
    with pytest.raises(TextureError, match="provider") as caught:
        TextureEngine(RemoteStableDiffusionProvider(transport), tmp_path / "out").generate(
            "red stone", mesh_path
        )
    assert caught.value.__cause__ is error
    assert not (tmp_path / "out").exists()


def test_bad_remote_response_is_not_saved(mesh_path, tmp_path):
    with pytest.raises(TextureValidationError):
        TextureEngine(
            RemoteStableDiffusionProvider(Mock(return_value=b"not PNG")), tmp_path / "out"
        ).generate("red stone", mesh_path)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("provider", [
    DeterministicTestProvider(), RemoteStableDiffusionProvider(Mock())
])
def test_depth_is_explicitly_unsupported(provider, mesh_path, tmp_path):
    with pytest.raises(TextureValidationError, match="not implemented"):
        TextureEngine(provider, tmp_path / "out").generate(
            "red stone", mesh_path, depth_map_path=tmp_path / "depth.png"
        )
    assert not (tmp_path / "out").exists()


def test_write_failure_is_meaningful(mesh_path, tmp_path):
    output = tmp_path / "file"
    output.write_text("preserve", encoding="utf-8")
    with pytest.raises(TextureError, match="save"):
        TextureEngine(DeterministicTestProvider(), output).generate("red stone", mesh_path)
    assert output.read_text(encoding="utf-8") == "preserve"


def test_existing_output_is_not_overwritten(mesh_path, tmp_path, monkeypatch):
    monkeypatch.setattr("texture_engine.engine.uuid4", lambda: Mock(hex="fixed"))
    output = tmp_path / "test_texture_fixed.png"
    output.write_bytes(b"preserve")
    with pytest.raises(TextureError, match="save"):
        TextureEngine(DeterministicTestProvider(), tmp_path).generate("red stone", mesh_path)
    assert output.read_bytes() == b"preserve"


def test_png_rgb_and_rgba_are_accepted(png):
    assert validate_png(png) == (8, 8)
    assert validate_png(make_png(color=6, pixels=b"\x00\x01\x02\x03\xff")) == (1, 1)


@pytest.mark.parametrize("data", [
    b"", b"not PNG", None, b"x" * (16 * 1024 * 1024 + 1),
    PNG_SIGNATURE, make_png()[:-1], make_png() + b"trailing",
    make_png(width=0), make_png(width=2049), make_png(color=0),
    make_png(pixels=b"\x00"), make_png(pixels=b"\x05\x01\x02\x03"),
])
def test_invalid_png_is_rejected(data):
    with pytest.raises(TextureValidationError):
        validate_png(data)


def test_crc_corruption_is_rejected(png):
    damaged = bytearray(png)
    damaged[29] ^= 1
    with pytest.raises(TextureValidationError, match="checksum"):
        validate_png(bytes(damaged))


def test_compressed_expansion_is_bounded():
    with pytest.raises(TextureValidationError, match="size"):
        validate_png(make_png(pixels=b"\x00" * 100_000))


def test_remote_requires_explicit_transport():
    with pytest.raises(TypeError, match="transport"):
        RemoteStableDiffusionProvider(None)
