"""
Utility helpers for the Teachable Machine Frontend.
Provides image processing helpers and formatting functions.
"""
import io
from PIL import Image


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """
    Converts raw image bytes to a PIL Image object.
    Useful for displaying preview thumbnails from uploaded/captured frames.
    """
    return Image.open(io.BytesIO(image_bytes))


def resize_for_preview(pil_image: Image.Image, max_size: int = 280) -> Image.Image:
    """
    Resizes a PIL Image proportionally so it fits within max_size x max_size,
    for clean thumbnail previews in the UI without distortion.
    """
    pil_image.thumbnail((max_size, max_size), Image.LANCZOS)
    return pil_image


def sanitize_class_name(name: str) -> str:
    """
    Strips all characters that are not alphanumeric, hyphens, or underscores.
    This must match the backend's sanitization logic exactly to ensure consistent
    directory naming between the frontend inputs and the backend's saved folder paths.
    """
    return "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()


def confidence_to_color(confidence: float) -> str:
    """
    Returns a CSS hex color code based on confidence value.
    - High confidence (> 0.75): vibrant green
    - Medium confidence (0.4 - 0.75): amber
    - Low confidence (< 0.4): red
    """
    if confidence >= 0.75:
        return "#10b981"  # emerald green
    elif confidence >= 0.4:
        return "#f59e0b"  # amber
    else:
        return "#ef4444"  # red


def format_pct(value: float) -> str:
    """Formats a 0.0-1.0 float as a percentage string e.g. '87.4%'."""
    return f"{value * 100:.1f}%"
