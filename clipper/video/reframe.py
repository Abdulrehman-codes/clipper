"""Reframe a 16:9 clip to 9:16 by following the dominant face (§4.5).

MVP approach:
  1. Sample face positions per frame with MediaPipe face detection.
  2. Build a smoothed (EMA) horizontal crop-center track, velocity-capped so the
     crop never snaps -- "smoothing matters more than accuracy" (§4.5).
  3. Re-encode with a moving crop via an ffmpeg sendcmd/expression OR, when the
     track is near-static, a single crop. Fallback to blur-fill/letterbox when no
     face is found.

Horizontal (16:9) output is just the cut, no crop (caller handles that).

MediaPipe + OpenCV are optional extras; if missing we fall back to a static
center crop so the pipeline still produces a 9:16 file.
"""

from __future__ import annotations

from pathlib import Path

from ..config import ReframeConfig
from ..ffmpeg_utils import probe_dimensions, run


def _aspect_to_ratio(aspect: str) -> float:
    w, h = aspect.split(":")
    return float(w) / float(h)


def _sample_face_centers(src: str, cfg: ReframeConfig, sample_fps: float = 5.0):
    """Return list of (t_seconds, cx_fraction) face-center samples, or [] if MediaPipe
    is unavailable / no faces. cx_fraction is 0..1 across frame width."""
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except Exception:
        return []

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))

    samples: list[tuple[float, float]] = []
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=cfg.detection_confidence
    )
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = detector.process(rgb)
                if res.detections:
                    # Dominant face = highest detection score.
                    det = max(res.detections, key=lambda d: d.score[0])
                    box = det.location_data.relative_bounding_box
                    cx = box.xmin + box.width / 2.0
                    samples.append((idx / fps, min(max(cx, 0.0), 1.0)))
            idx += 1
    finally:
        cap.release()
        detector.close()
    return samples


def _smooth_track(samples, cfg: ReframeConfig, default: float = 0.5) -> list[tuple[float, float]]:
    """EMA-smooth + velocity-cap the crop-center track (§4.5)."""
    if not samples:
        return []
    out: list[tuple[float, float]] = []
    ema = samples[0][1]
    prev = ema
    for t, cx in samples:
        ema = cfg.smoothing * cx + (1 - cfg.smoothing) * ema
        delta = ema - prev
        if abs(delta) > cfg.max_velocity:
            ema = prev + cfg.max_velocity * (1 if delta > 0 else -1)
        prev = ema
        out.append((t, ema))
    return out


def _crop_expr_x(track, crop_w: int, src_w: int) -> str:
    """Build a piecewise ffmpeg expression for crop x over time from the track.

    Uses nested if(lt(t,..)) so the crop center follows the smoothed face track.
    Falls back to a constant if the track is empty."""
    max_x = max(0, src_w - crop_w)
    if not track:
        return str(max_x // 2)

    def clamp_x(cx: float) -> int:
        x = int(round(cx * src_w - crop_w / 2))
        return max(0, min(max_x, x))

    # Build from the last sample backwards so earlier conditions take priority.
    expr = str(clamp_x(track[-1][1]))
    for t, cx in reversed(track[:-1]):
        expr = f"if(lt(t,{t:.3f}),{clamp_x(cx)},{expr})"
    return expr


def reframe_to_vertical(
    src: str | Path,
    dst: str | Path,
    cfg: ReframeConfig,
    encode,
) -> Path:
    """Produce a 9:16 crop following the dominant face."""
    src = str(src)
    dst = Path(dst)
    src_w, src_h = probe_dimensions(src)

    target = _aspect_to_ratio(cfg.target_vertical)  # 9/16 = 0.5625
    # Crop width that yields target aspect at full source height.
    crop_w = int(round(src_h * target))

    if crop_w <= src_w:
        # Source is wide enough to crop a vertical slice out of it.
        samples = _sample_face_centers(src, cfg)
        track = _smooth_track(samples, cfg)
        x_expr = _crop_expr_x(track, crop_w, src_w)
        # Even dimensions required by libx264.
        crop_w -= crop_w % 2
        vf = f"crop={crop_w}:ih:{x_expr}:0,scale=1080:1920:flags=lanczos"
        run([
            "ffmpeg", "-y", "-i", src,
            "-vf", vf,
            "-c:v", encode.video_codec, "-crf", str(encode.crf), "-preset", encode.preset,
            "-c:a", "copy", "-movflags", "+faststart",
            str(dst),
        ])
        return dst

    # Source too narrow to crop -> fill (blur or letterbox) into 9:16 (§4.5).
    if cfg.fallback_fill == "blur":
        vf = (
            "split[a][b];"
            "[a]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,boxblur=40:1[bg];"
            "[b]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )
    else:  # letterbox
        vf = (
            "scale=1080:1920:force_original_aspect_ratio=decrease,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
        )
    run([
        "ffmpeg", "-y", "-i", src,
        "-filter_complex", vf,
        "-c:v", encode.video_codec, "-crf", str(encode.crf), "-preset", encode.preset,
        "-c:a", "copy", "-movflags", "+faststart",
        str(dst),
    ])
    return dst
