"""Upload handling: size limit, type check, integrity check, decoding, PNG encoding."""
from __future__ import annotations

import base64
import io

import numpy as np
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from backend.app.core.config import Settings
from backend.app.core.errors import FileTooLargeError, InvalidImageError, UnsupportedMediaTypeError

CHUNK_SIZE = 1024 * 1024
_FORMAT_TO_TYPE = {"PNG": "image/png", "JPEG": "image/jpeg"}


async def read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Read the upload in 1 MB chunks and stop as soon as it exceeds ``max_bytes`` (413)."""
    buf = bytearray()
    while chunk := await upload.read(CHUNK_SIZE):
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise FileTooLargeError(f"File exceeds the {max_bytes // (1024 * 1024)} MB limit")
    return bytes(buf)


def check_declared_type(upload: UploadFile, settings: Settings) -> None:
    """415 if the client declares a type we do not accept (the real format is checked after decoding)."""
    if upload.content_type not in settings.allowed_content_types:
        raise UnsupportedMediaTypeError(
            f"Unsupported file type '{upload.content_type}'. Upload a PNG or JPEG image."
        )


def decode_grayscale(data: bytes, settings: Settings) -> np.ndarray:
    """Verify and decode image bytes to a uint8 grayscale array."""
    if not data:
        raise InvalidImageError("The uploaded file is empty")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()  # detects truncated / corrupted files
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError("The file is not a valid image or is corrupted") from exc
    if fmt not in _FORMAT_TO_TYPE:
        raise UnsupportedMediaTypeError(f"Image format '{fmt}' is not supported. Upload a PNG or JPEG image.")
    try:
        with Image.open(io.BytesIO(data)) as img:  # verify() leaves the object unusable, so reopen
            w, h = img.size
            if w * h > settings.max_image_pixels:
                raise InvalidImageError(f"Image is too large ({w}x{h} pixels)")
            if min(w, h) < settings.min_image_side:
                raise InvalidImageError(
                    f"Image is too small ({w}x{h}); each side must be at least {settings.min_image_side} px"
                )
            gray = np.asarray(img.convert("L"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise InvalidImageError("The image could not be decoded") from exc
    return gray


def png_base64(array: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")
