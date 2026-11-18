"""HTTPS transport for an AUTOMATIC1111-compatible txt2img API.

Use RemoteStableDiffusionProvider.from_env() with these environment variables:
SD_BASE_URL (HTTPS, optional path prefix), SD_USERNAME, SD_PASSWORD, SD_MODEL
(installed checkpoint), SD_SEED (default -1), SD_WIDTH and SD_HEIGHT (512),
and SD_TIMEOUT (120 seconds, per blocking network operation).
"""

import base64
import binascii
import http.client
import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from texture_engine.validation import (
    MAX_DIMENSION,
    MAX_PNG_BYTES,
    TextureError,
    TextureValidationError,
    validate_png,
)


MAX_BASE64_BYTES = 4 * ((MAX_PNG_BYTES + 2) // 3)
MAX_RESPONSE_BYTES = MAX_BASE64_BYTES + 1024 * 1024


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class StableDiffusionHTTPTransport:
    """Send one txt2img request per call using a snapshot of the environment."""

    def __init__(self):
        base_url = os.environ.get("SD_BASE_URL", "").strip()
        try:
            url = urlsplit(base_url)
            valid_url = (
                url.scheme == "https" and url.hostname and url.port != 0
                and url.username is None and url.password is None
                and not any(c.isspace() or ord(c) < 32 for c in base_url)
                and not any(c in base_url for c in "?#\\")
            )
        except ValueError:
            valid_url = False
        if not valid_url:
            raise TextureValidationError("SD_BASE_URL must be an HTTPS base URL.")
        self._endpoint = base_url.rstrip("/") + "/sdapi/v1/txt2img"

        username = os.environ.get("SD_USERNAME", "")
        password = os.environ.get("SD_PASSWORD", "")
        if not username.strip() or not password.strip() or ":" in username:
            raise TextureValidationError("Set valid SD_USERNAME and SD_PASSWORD credentials.")
        credentials = base64.b64encode(f"{username}:{password}".encode("utf-8"))
        self._authorization = "Basic " + credentials.decode("ascii")

        model = os.environ.get("SD_MODEL", "").strip()
        if not model:
            raise TextureValidationError("SD_MODEL must name an installed checkpoint.")
        try:
            seed = int(os.environ.get("SD_SEED", "-1"))
            width = int(os.environ.get("SD_WIDTH", "512"))
            height = int(os.environ.get("SD_HEIGHT", "512"))
            timeout = float(os.environ.get("SD_TIMEOUT", "120"))
        except ValueError:
            raise TextureValidationError("SD_SEED, SD_WIDTH, SD_HEIGHT and SD_TIMEOUT must be numeric.") from None
        if not -1 <= seed <= 2**32 - 1:
            raise TextureValidationError("SD_SEED must be -1 or an unsigned 32-bit integer.")
        if not all(8 <= size <= MAX_DIMENSION and size % 8 == 0 for size in (width, height)):
            raise TextureValidationError("SD_WIDTH and SD_HEIGHT must be multiples of 8 from 8 to 2048.")
        if not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise TextureValidationError("SD_TIMEOUT must be greater than 0 and at most 3600 seconds.")
        self._timeout = timeout
        self._size = width, height
        self._options = {
            "seed": seed,
            "width": width,
            "height": height,
            "batch_size": 1,
            "n_iter": 1,
            "send_images": True,
            "save_images": False,
            "override_settings": {"sd_model_checkpoint": model},
            "override_settings_restore_afterwards": True,
        }
        self._opener = build_opener(_NoRedirects())

    def __call__(self, request):
        http_request = Request(
            self._endpoint,
            data=json.dumps(dict(self._options, prompt=request.prompt)).encode("utf-8"),
            headers={
                "Authorization": self._authorization,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
            method="POST",
        )
        try:
            with self._opener.open(http_request, timeout=self._timeout) as response:
                if response.headers.get_content_type() != "application/json":
                    raise TextureValidationError("Stable Diffusion response must be JSON.")
                body = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            exc.close()
            raise TextureError(f"Stable Diffusion HTTP request failed (status {exc.code}).") from None
        except TimeoutError:
            raise TextureError("Stable Diffusion request timed out.") from None
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TextureError("Stable Diffusion request timed out.") from None
            raise TextureError("Stable Diffusion network request failed.") from None
        except (OSError, http.client.HTTPException):
            raise TextureError("Stable Diffusion network request failed.") from None
        if len(body) > MAX_RESPONSE_BYTES:
            raise TextureValidationError("Stable Diffusion response exceeds the size limit.")
        try:
            payload = json.loads(body)
        except (ValueError, RecursionError):
            raise TextureValidationError("Stable Diffusion returned malformed JSON.") from None
        images = payload.get("images") if isinstance(payload, dict) else None
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise TextureValidationError("Stable Diffusion must return one base64-encoded PNG.")
        if len(images[0]) > MAX_BASE64_BYTES:
            raise TextureValidationError("Stable Diffusion image exceeds the size limit.")
        try:
            png = base64.b64decode(images[0], validate=True)
        except (ValueError, binascii.Error):
            raise TextureValidationError("Stable Diffusion returned invalid base64.") from None
        if validate_png(png) != self._size:
            raise TextureValidationError("Stable Diffusion PNG dimensions differ from the request.")
        return png
