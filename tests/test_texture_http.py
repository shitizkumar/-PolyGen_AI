"""Day 2 transport and atomic publication tests. All HTTP calls are mocked."""

import base64
from email.message import Message
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, Mock
from urllib.error import HTTPError, URLError

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
from texture_engine import http_transport


@pytest.fixture
def config(monkeypatch):
    values = {
        "SD_BASE_URL": "https://sd.example.test/prefix/",
        "SD_USERNAME": "operator",
        "SD_PASSWORD": "mock-secret",
        "SD_MODEL": "installed-checkpoint",
        "SD_SEED": "123",
        "SD_WIDTH": "8",
        "SD_HEIGHT": "8",
        "SD_TIMEOUT": "2.5",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def request_data():
    return TextureRequest("red stone", Path("private-mesh.obj"))


@pytest.fixture
def server(monkeypatch):
    opener = MagicMock()
    monkeypatch.setattr(http_transport, "build_opener", Mock(return_value=opener))
    response = MagicMock()
    response.headers = Message()
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    opener.open.return_value.__enter__.return_value = response
    return opener, response


def test_http_contract(config, server, request_data):
    opener, response = server
    png = DeterministicTestProvider().generate(request_data)
    response.read.return_value = json.dumps({"images": [base64.b64encode(png).decode()]}).encode()
    assert RemoteStableDiffusionProvider.from_env().generate(request_data) == png
    request = opener.open.call_args.args[0]
    assert request.full_url == "https://sd.example.test/prefix/sdapi/v1/txt2img"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Basic " + base64.b64encode(
        b"operator:mock-secret"
    ).decode()
    assert json.loads(request.data) == {
        "prompt": "red stone", "width": 8, "height": 8, "seed": 123,
        "batch_size": 1, "n_iter": 1, "send_images": True, "save_images": False,
        "override_settings": {"sd_model_checkpoint": "installed-checkpoint"},
        "override_settings_restore_afterwards": True,
    }
    assert opener.open.call_args.kwargs == {"timeout": 2.5}
    response.read.assert_called_once_with(http_transport.MAX_RESPONSE_BYTES + 1)


@pytest.mark.parametrize(("key", "value"), [
    ("SD_BASE_URL", "http://sd.example.test"),
    ("SD_BASE_URL", "https://user:pass@sd.example.test"),
    ("SD_PASSWORD", ""), ("SD_MODEL", ""), ("SD_SEED", "-2"),
    ("SD_WIDTH", "513"), ("SD_HEIGHT", "2056"), ("SD_TIMEOUT", "nan"),
])
def test_config_validation(config, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(TextureValidationError, match=key):
        RemoteStableDiffusionProvider.from_env()


@pytest.mark.parametrize(("error", "message"), [
    (HTTPError("https://sd.example.test", 401, "mock-secret", {}, io.BytesIO()), "401"),
    (HTTPError("https://sd.example.test", 429, "mock-secret", {}, io.BytesIO()), "429"),
    (HTTPError("https://sd.example.test", 503, "mock-secret", {}, io.BytesIO()), "503"),
    (URLError("mock-secret"), "network"),
    (URLError(TimeoutError("mock-secret")), "timed out"),
    (TimeoutError("mock-secret"), "timed out"),
])
def test_http_failures_are_sanitized(config, server, request_data, error, message):
    opener, _ = server
    opener.open.side_effect = error
    with pytest.raises(TextureError, match=message) as caught:
        RemoteStableDiffusionProvider.from_env().generate(request_data)
    assert "mock-secret" not in str(caught.value)
    assert caught.value.__suppress_context__
    opener.open.assert_called_once()


def test_redirects_are_not_followed():
    assert http_transport._NoRedirects().redirect_request(
        None, None, 302, "Found", {}, "https://another.example.test"
    ) is None


@pytest.mark.parametrize(("body", "message"), [
    (b"{", "malformed JSON"),
    (b'{"images":[]}', "one base64"),
    (b'{"images":["?"]}', "invalid base64"),
    (b'{"images":["eA=="]}', "not a PNG"),
])
def test_invalid_responses(config, server, request_data, body, message):
    _, response = server
    response.read.return_value = body
    with pytest.raises(TextureValidationError, match=message):
        RemoteStableDiffusionProvider.from_env().generate(request_data)


def test_response_limit(config, server, request_data, monkeypatch):
    monkeypatch.setattr(http_transport, "MAX_RESPONSE_BYTES", 16)
    _, response = server
    response.read.return_value = b"x" * 17
    with pytest.raises(TextureValidationError, match="size limit"):
        RemoteStableDiffusionProvider.from_env().generate(request_data)
    response.read.assert_called_once_with(17)


def test_dimension_mismatch(config, server, request_data, monkeypatch):
    monkeypatch.setenv("SD_WIDTH", "16")
    _, response = server
    png = DeterministicTestProvider().generate(request_data)
    response.read.return_value = json.dumps({"images": [base64.b64encode(png).decode()]}).encode()
    with pytest.raises(TextureValidationError, match="dimensions differ"):
        RemoteStableDiffusionProvider.from_env().generate(request_data)


def test_failed_atomic_publication_cleans_up(tmp_path, monkeypatch):
    mesh = tmp_path / "mesh.obj"
    trimesh.creation.box().export(mesh)
    directory = tmp_path / "textures"
    monkeypatch.setattr("texture_engine.engine.os.link", Mock(side_effect=OSError("publish failed")))
    with pytest.raises(TextureError, match="save"):
        TextureEngine(DeterministicTestProvider(), directory).generate("red stone", mesh)
    assert list(directory.iterdir()) == []
