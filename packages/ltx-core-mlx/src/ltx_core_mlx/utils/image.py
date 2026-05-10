"""Image and video preparation utilities for VAE encoding."""

from __future__ import annotations

import math
import subprocess

import mlx.core as mx
import numpy as np
from PIL import Image

from ltx_core_mlx.utils.ffmpeg import find_ffmpeg

# Match upstream ``ltx_pipelines.utils.args.DEFAULT_IMAGE_CRF``. Round-tripping
# the input image through libx264 at this CRF brings it close to the LTX-2
# training distribution (which is trained on real video frames carrying H.264
# compression artefacts), preventing the model from over-reacting to pristine
# PNG/JPEG textures during I2V conditioning.
DEFAULT_IMAGE_CRF = 33


def apply_crf_compression(image: Image.Image, crf: int) -> Image.Image:
    """Round-trip a PIL image through libx264 at the given CRF.

    Mirrors upstream ``ltx_pipelines.utils.media_io.preprocess`` (PyAV-based)
    using an ``ffmpeg`` subprocess pipeline:

    1. Encode raw RGB pixels into a 1-frame H.264 mp4 with the given CRF.
    2. Decode the resulting mp4 back to raw RGB.

    The output is the same PIL image with H.264-style compression artefacts
    (blocking, ringing, slight chroma shift). ``crf == 0`` is a no-op.

    Args:
        image: Input PIL image (RGB mode).
        crf: Constant Rate Factor (0 = no compression / passthrough,
            higher = more compression). Default upstream is 33.

    Returns:
        New PIL image with the round-trip applied.
    """
    if crf == 0:
        return image
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    # H.264 requires even dimensions. Pad to next even pixel; we crop back.
    pad_w = width + (width & 1)
    pad_h = height + (height & 1)
    if (pad_w, pad_h) != (width, height):
        padded = Image.new("RGB", (pad_w, pad_h), (0, 0, 0))
        padded.paste(image, (0, 0))
        image_for_encode = padded
    else:
        image_for_encode = image

    raw_in = np.asarray(image_for_encode, dtype=np.uint8).tobytes()
    ffmpeg = find_ffmpeg()

    # Encode raw RGB → H.264 mp4 in memory.
    encode_cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{pad_w}x{pad_h}",
        "-r",
        "1",
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(int(crf)),
        "-frames:v",
        "1",
        "-f",
        "mp4",
        "-movflags",
        "frag_keyframe+empty_moov",
        "pipe:1",
    ]
    enc = subprocess.run(encode_cmd, input=raw_in, capture_output=True, timeout=60)
    if enc.returncode != 0:
        raise RuntimeError(f"ffmpeg CRF encode failed: {enc.stderr.decode(errors='ignore')}")

    # Decode mp4 → raw RGB.
    decode_cmd = [
        ffmpeg,
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    dec = subprocess.run(decode_cmd, input=enc.stdout, capture_output=True, timeout=60)
    if dec.returncode != 0:
        raise RuntimeError(f"ffmpeg CRF decode failed: {dec.stderr.decode(errors='ignore')}")

    arr = np.frombuffer(dec.stdout, dtype=np.uint8).reshape(pad_h, pad_w, 3)
    out = Image.fromarray(arr, mode="RGB")
    if (pad_w, pad_h) != (width, height):
        out = out.crop((0, 0, width, height))
    return out


def prepare_image_for_encoding(
    image: Image.Image | str,
    height: int,
    width: int,
    crf: int = DEFAULT_IMAGE_CRF,
) -> mx.array:
    """Load and prepare an image for VAE encoding.

    1. Optionally round-trips the image through libx264 at ``crf`` to match
       the LTX-2 training distribution (upstream-iso). Pass ``crf=0`` to skip.
    2. Aspect-preserving resize + center crop to ``(height, width)``.
    3. Normalize to ``[-1, 1]``, return as ``(1, 3, H, W)`` bfloat16.

    Args:
        image: PIL Image or path to image file.
        height: Target height.
        width: Target width.
        crf: H.264 CRF for the round-trip (default 33; pass 0 to skip).

    Returns:
        mx.array of shape (1, 3, H, W) in [-1, 1] range, bfloat16.
    """
    if isinstance(image, str):
        image = Image.open(image)

    image = image.convert("RGB")

    # CRF round-trip (upstream-iso). Applied BEFORE resize so the artefacts
    # land on full-resolution pixels — matching how upstream's preprocess()
    # is invoked in load_image_and_preprocess (encode → decode → resize).
    if crf and crf > 0:
        image = apply_crf_compression(image, crf)

    # Aspect-ratio-preserving resize + center crop (matches reference)
    src_w, src_h = image.size
    scale = max(height / src_h, width / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    # Center crop to target size
    crop_left = (new_w - width) // 2
    crop_top = (new_h - height) // 2
    image = image.crop((crop_left, crop_top, crop_left + width, crop_top + height))

    # HWC uint8 -> float32 -> [-1, 1]
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0

    # HWC -> CHW -> BCHW
    tensor = mx.array(arr).transpose(2, 0, 1)[None, ...]
    return tensor.astype(mx.bfloat16)


def load_video_frames(
    video_path: str,
    height: int,
    width: int,
    num_frames: int,
) -> mx.array:
    """Load video frames via ffmpeg as a tensor for VAE encoding.

    Args:
        video_path: Path to the video file.
        height: Frame height in pixels.
        width: Frame width in pixels.
        num_frames: Number of frames to read.

    Returns:
        Video tensor of shape (1, 3, F, H, W) in [-1, 1] range, bfloat16.

    Raises:
        RuntimeError: If ffmpeg fails to read the video.
    """
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg,
        "-i",
        video_path,
        "-vframes",
        str(num_frames),
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to read video: {result.stderr.decode()}")

    raw = result.stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
    # Normalize to [-1, 1]
    frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
    # FHWC -> BCFHW: (F, H, W, 3) -> (3, F, H, W) -> (1, 3, F, H, W)
    tensor = mx.array(frames).transpose(3, 0, 1, 2)[None, ...]
    return tensor.astype(mx.bfloat16)
