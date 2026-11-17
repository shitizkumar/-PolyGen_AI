"""Bounded PNG validation without an additional imaging dependency.

Day 1 accepts static, non-interlaced, 8-bit RGB/RGBA PNGs only. Other valid
PNG encodings are deliberately rejected until an imaging backend is selected.
"""

import struct
import zlib


class TextureError(RuntimeError):
    """Texture generation or persistence failed."""


class TextureValidationError(ValueError):
    """A texture request or provider response is invalid."""


MAX_PNG_BYTES = 16 * 1024 * 1024
MAX_DIMENSION = 2048
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def validate_png(data: bytes) -> tuple[int, int]:
    """Check structure, CRCs, dimensions and bounded decompression before saving."""
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_PNG_BYTES:
        raise TextureValidationError("PNG must be bytes, at most 16 MiB.")
    if not data.startswith(PNG_SIGNATURE):
        raise TextureValidationError("Provider response is not a PNG.")
    offset = len(PNG_SIGNATURE)
    seen: set[bytes] = set()
    previous = None
    compressed = bytearray()
    width = height = channels = 0
    while offset < len(data):
        if offset + 12 > len(data):
            raise TextureValidationError("Truncated PNG chunk.")
        length = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4:offset + 8]
        end = offset + 12 + length
        if end > len(data):
            raise TextureValidationError("Truncated PNG payload.")
        payload = data[offset + 8:end - 4]
        crc = struct.unpack_from(">I", data, end - 4)[0]
        if zlib.crc32(kind + payload) != crc:
            raise TextureValidationError("PNG checksum mismatch.")
        if not all(65 <= value <= 90 or 97 <= value <= 122 for value in kind):
            raise TextureValidationError("Invalid PNG chunk type.")
        if kind[2] & 32:
            raise TextureValidationError("Invalid PNG reserved chunk bit.")
        if not seen and kind != b"IHDR":
            raise TextureValidationError("PNG must begin with IHDR.")
        if kind == b"IHDR":
            if seen or length != 13:
                raise TextureValidationError("Invalid PNG header.")
            width, height, depth, color, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", payload)
            )
            if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION):
                raise TextureValidationError("PNG dimensions must be between 1 and 2048.")
            if depth != 8 or color not in (2, 6) or any((compression, filtering, interlace)):
                raise TextureValidationError("Only non-interlaced 8-bit RGB/RGBA PNGs are supported.")
            channels = 3 if color == 2 else 4
        elif kind == b"IDAT":
            if b"IDAT" in seen and previous != b"IDAT":
                raise TextureValidationError("PNG IDAT chunks must be consecutive.")
            compressed.extend(payload)
        elif kind == b"IEND":
            if length or end != len(data) or b"IDAT" not in seen:
                raise TextureValidationError("Invalid PNG end chunk.")
        elif kind in (b"acTL", b"fcTL", b"fdAT") or not kind[0] & 32:
            raise TextureValidationError("Unsupported PNG chunk.")
        seen.add(kind)
        previous = kind
        offset = end
    if previous != b"IEND":
        raise TextureValidationError("PNG is missing IEND.")
    expected = height * (1 + width * channels)
    try:
        decoder = zlib.decompressobj()
        pixels = decoder.decompress(bytes(compressed), expected + 1)
    except zlib.error as exc:
        raise TextureValidationError("Invalid PNG compressed data.") from exc
    if len(pixels) != expected or not decoder.eof or decoder.unused_data:
        raise TextureValidationError("PNG pixel data has an invalid size or stream.")
    stride = 1 + width * channels
    if any(pixels[index] > 4 for index in range(0, expected, stride)):
        raise TextureValidationError("PNG contains an invalid scanline filter.")
    return width, height
