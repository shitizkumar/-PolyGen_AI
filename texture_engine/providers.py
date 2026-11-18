"""Explicit provider contracts, with no implicit fallback or GPU dependency."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import struct
from typing import Callable, Protocol
import zlib

from texture_engine.validation import PNG_SIGNATURE, TextureValidationError


@dataclass(frozen=True)
class TextureRequest:
    prompt: str
    mesh_path: Path
    # Reserved for future depth-conditioned providers. No depth rendering or
    # ControlNet implementation is provided by this foundation.
    depth_map_path: Path | None = None


@dataclass(frozen=True)
class TextureResult:
    texture_path: Path
    mesh_path: Path
    provider: str
    is_test_texture: bool
    width: int
    height: int


class TextureProvider(Protocol):
    name: str
    is_test_provider: bool
    supports_depth_conditioning: bool

    def generate(self, request: TextureRequest) -> bytes:
        """Return PNG bytes or raise; never write mesh or output files."""
        ...


def _reject_depth(request: TextureRequest) -> None:
    if request.depth_map_path is not None:
        raise TextureValidationError("Depth-conditioned ControlNet is not implemented.")


class DeterministicTestProvider:
    """Synthetic solid-color fixture derived from the prompt; NOT an AI texture."""

    name = "deterministic-test"
    is_test_provider = True
    supports_depth_conditioning = False

    def generate(self, request: TextureRequest) -> bytes:
        _reject_depth(request)
        color = sha256(request.prompt.encode("utf-8")).digest()[:3]

        def chunk(kind: bytes, payload: bytes) -> bytes:
            return (struct.pack(">I", len(payload)) + kind + payload
                    + struct.pack(">I", zlib.crc32(kind + payload)))

        return (
            PNG_SIGNATURE
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\x00" + color * 8) * 8))
            + chunk(b"IEND", b"")
        )


class RemoteStableDiffusionProvider:
    """Optional adapter for a caller-supplied remote SD transport.

    The transport owns endpoint configuration, authentication, timeouts,
    response size limits, and decoding its vendor's response into PNG bytes.
    No remote API dialect or ControlNet support is assumed. Failures propagate
    to the engine; this adapter never substitutes a synthetic texture.
    """

    name = "remote-stable-diffusion"
    is_test_provider = False
    supports_depth_conditioning = False

    def __init__(self, transport: Callable[[TextureRequest], bytes]):
        if not callable(transport):
            raise TypeError("A configured remote transport callable is required.")
        self._transport = transport

    @classmethod
    def from_env(cls) -> "RemoteStableDiffusionProvider":
        """Configure the AUTOMATIC1111 HTTPS transport from the environment."""
        from texture_engine.http_transport import StableDiffusionHTTPTransport

        return cls(StableDiffusionHTTPTransport())

    def generate(self, request: TextureRequest) -> bytes:
        _reject_depth(request)
        return self._transport(request)
