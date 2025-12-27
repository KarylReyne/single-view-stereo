# ============================================================================
#  NOTICE / DISCLAIMER (UNTESTED • EXPERIMENTAL • UNREVIEWED)
# ============================================================================
#  This Python file is provided in an "AS IS" and "AS AVAILABLE" state.
#  It is untested, experimental, and has not been reviewed. It may be
#  incomplete, incorrect, insecure, or otherwise unsuitable for any purpose.
#
#  USE AT YOUR OWN RISK.
#  By running, importing, modifying, or otherwise using this file, you accept
#  full responsibility for any outcomes, including (without limitation) data
#  loss, system instability, security incidents, downtime, financial loss, or
#  other direct or indirect damages.
#
#  NO WARRANTY.
#  To the fullest extent permitted by applicable law, the author(s) and any
#  contributor(s) disclaim all warranties, express or implied, including but
#  not limited to warranties of merchantability, fitness for a particular
#  purpose, non-infringement, accuracy, and absence of errors or defects.
#
#  LIMITATION OF LIABILITY.
#  To the fullest extent permitted by applicable law, in no event shall the
#  author(s) or contributor(s) be liable for any claim, damages, or other
#  liability, whether in an action of contract, tort, or otherwise, arising
#  from, out of, or in connection with the file or the use or other dealings
#  in the file, including any special, incidental, consequential, exemplary,
#  or punitive damages, even if advised of the possibility of such damages.
#
#  SAFETY / SECURITY NOTE.
#  Do not run this file on production systems or with sensitive data. Review
#  the code, dependencies, permissions, and inputs carefully before execution.
#
#  If you do not agree to these terms, do not use this file.
# ============================================================================

"""Stereo Image Evaluation Desktop App (PySide6) – single-file implementation.

User workflow
-------------
1) Choose an input root folder.
2) Click "Scan" to discover collection folders.
3) Choose which metrics (algorithms) you want to visualize via the checkbox list.
4) Click "Run" to compute ALL metrics for ALL collections (batch mode).
5) Select a collection on the left to see images, metadata, and a bar chart:
   - First bar: Average score over the selected metrics (ignoring unavailable)
   - Following bars: Scores per selected metric

Folder structure per collection
------------------------------
<root>/.../<collection_name>/
  left.jpg
  right.jpg
  generated.jpg
  meta.json

Dependencies
------------
Required:
- PySide6
- numpy

Optional (enables more metrics / faster execution):
- opencv-python (cv2)
- scipy
- torch, torchvision
- lpips
- piq (adds DISTS, and sometimes PieAPP)
- open_clip_torch OR transformers (CLIP)
- timm OR transformers (DINO/DINOv2 similarity)

Notes
-----
- Learned metrics (LPIPS/VGG/CLIP/DISTS/DINO) are implemented as wrappers.
  If dependencies/models are missing, they are shown as "Unavailable".
- Stereo consistency metric is included (uses left): Disparity Consistency EPE (StereoSGBM).
  Treat it as an optional "extra" metric (select via checkbox).

Must have
-----
pip install PySide6 numpy

Optional
-----
pip install opencv-python scipy torch torchvision lpips piq open_clip_torch timm transformers pillow

Run
---
python stereo_evaluation_app.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets


# =============================================================================
# Optional dependencies
# =============================================================================
#
# The app is designed to be useful even with minimal dependencies (PySide6 + NumPy).
# Advanced metrics and faster image operations become available if optional packages
# are installed. Each metric checks availability at runtime and will report a clear
# reason if it cannot run.
#
# Important: keep import failures non-fatal; the UI should continue to work.
# =============================================================================

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

try:
    import scipy  # type: ignore
    from scipy import ndimage  # type: ignore
    from scipy.fftpack import dct as scipy_dct  # type: ignore

    _HAS_SCIPY = True
except Exception:
    scipy = None
    ndimage = None
    scipy_dct = None
    _HAS_SCIPY = False

try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    _HAS_TORCH = True
except Exception:
    torch = None
    F = None
    _HAS_TORCH = False

try:
    import torchvision  # type: ignore

    _HAS_TORCHVISION = True
except Exception:
    torchvision = None
    _HAS_TORCHVISION = False


# Global lock to serialize heavyweight torch model usage across worker threads.
#
# Why this exists:
# - Some models are large and expensive to run in parallel.
# - In multi-threaded batch runs, multiple workers might attempt to load models
#   at the same time, causing excessive memory use or CUDA contention.
# - Serializing the learned metrics avoids many practical stability issues.
_TORCH_LOCK = QtCore.QMutex()


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class CollectionItem:
    """Represents one collection folder with the expected file names."""

    name: str
    path: str
    left_path: str
    right_path: str
    gen_path: str
    meta_path: str


@dataclass
class MetricResult:
    """Holds both a raw value and a normalized score in [0..1] for charting."""

    key: str
    display_name: str
    description: str
    raw_value: Optional[float]
    score_0_1: Optional[float]
    available: bool
    details: str = ""


# =============================================================================
# Image IO & basic processing
# =============================================================================

def _read_image_rgb(path: str) -> np.ndarray:
    """Read an image file and return an RGB uint8 array of shape (H, W, 3).

    Preferred backend is OpenCV for speed and robustness. If OpenCV is not
    available, fall back to QImage.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    # Fast path: OpenCV
    if _HAS_CV2:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Fallback: Qt image reader
    qimg = QtGui.QImage(path)
    if qimg.isNull():
        raise ValueError(f"Failed to read image via QImage: {path}")

    # Ensure predictable memory layout: RGB888
    qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    w, h = qimg.width(), qimg.height()

    # QImage.bits() exposes the internal buffer; copy it so we own the memory.
    ptr = qimg.bits()
    ptr.setsize(h * w * 3)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3)).copy()
    return arr


def _to_float01(img_u8_rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB image to float32 in [0..1]."""
    return (img_u8_rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _rgb_to_gray(img_f01: np.ndarray) -> np.ndarray:
    """Convert float RGB in [0..1] to grayscale using ITU-R BT.709 coefficients."""
    return (
        0.2126 * img_f01[..., 0] + 0.7152 * img_f01[..., 1] + 0.0722 * img_f01[..., 2]
    ).astype(np.float32)


def _gaussian_blur(img: np.ndarray, sigma: float = 1.5, ksize: int = 11) -> np.ndarray:
    """Gaussian blur with progressively cheaper fallbacks.

    Order of preference:
    1) OpenCV (fast, well-tested)
    2) SciPy ndimage
    3) Pure NumPy separable convolution (slow, but avoids extra dependencies)
    """

    if _HAS_CV2:
        return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    if _HAS_SCIPY:
        return ndimage.gaussian_filter(img, sigma=sigma, mode="reflect")

    # Pure numpy fallback (separable Gaussian kernel)
    rad = ksize // 2
    x = np.arange(-rad, rad + 1, dtype=np.float32)
    g = np.exp(-(x * x) / (2.0 * sigma * sigma))
    g /= g.sum()

    def _convolve1d(a: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
        # Reflect padding to approximate typical image boundary handling.
        pad = len(kernel) // 2
        a_pad = np.pad(a, [(pad, pad) if i == axis else (0, 0) for i in range(a.ndim)], mode="reflect")
        out = np.zeros_like(a, dtype=np.float32)

        # Slide a 1D window along the selected axis.
        it = [slice(None)] * a.ndim
        for i in range(a.shape[axis]):
            it_in = it.copy()
            it_in[axis] = slice(i, i + 2 * pad + 1)
            window = a_pad[tuple(it_in)]

            out_it = it.copy()
            out_it[axis] = i
            out[tuple(out_it)] = np.tensordot(window, kernel, axes=([axis], [0]))
        return out

    # Handle grayscale vs RGB separately.
    if img.ndim == 2:
        tmp = _convolve1d(img, g, axis=0)
        return _convolve1d(tmp, g, axis=1)

    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        tmp = _convolve1d(img[..., c], g, axis=0)
        out[..., c] = _convolve1d(tmp, g, axis=1)
    return out


def _downsample2(img: np.ndarray) -> np.ndarray:
    """Downsample image by factor 2 using area averaging.

    MS-SSIM and VIFp approximate computations repeatedly downsample; this is a
    dependency-free approach (OpenCV path uses INTER_AREA for quality).
    """

    if _HAS_CV2:
        h, w = img.shape[:2]
        return cv2.resize(img, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]
    h2, w2 = max(1, h // 2), max(1, w // 2)

    # Ensure even spatial dimensions by trimming the last row/col if needed.
    img = img[: h2 * 2, : w2 * 2]

    if img.ndim == 2:
        return img.reshape(h2, 2, w2, 2).mean(axis=(1, 3))

    return img.reshape(h2, 2, w2, 2, img.shape[2]).mean(axis=(1, 3))


def _safe_mean(x: np.ndarray) -> float:
    """Mean with empty-array guard."""
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def _clamp01(x: float) -> float:
    """Clamp scalar to [0..1]."""
    return float(np.clip(x, 0.0, 1.0))


def _smoothstep(x: float) -> float:
    """Cubic smoothstep, assuming x in [0..1] (clamped internally)."""
    x = _clamp01(x)
    return float(x * x * (3.0 - 2.0 * x))


# =============================================================================
# Metric base class + registry
# =============================================================================

class Metric:
    """Abstract metric interface.

    Each metric produces:
    - raw value: in its natural units (error, similarity, dB, etc.)
    - score_0_1: normalized score for consistent charting (higher is better)

    Availability:
    - Metrics should be defensive and declare missing dependencies.
    - UI can hide unavailable metrics from the selection list.
    """

    key: str = ""
    display_name: str = ""
    description: str = ""

    # A rough guidance score for default/recommended selections.
    suitability: int = 3

    # Marks stereo-specific extra metrics that use the left image as context.
    is_stereo_extra: bool = False

    def is_available(self) -> Tuple[bool, str]:
        """Return (available, reason_if_unavailable)."""
        return True, ""

    def compute_raw(
        self,
        left_rgb_u8: np.ndarray,
        right_gt_rgb_u8: np.ndarray,
        right_gen_rgb_u8: np.ndarray,
        meta: Dict[str, Any],
    ) -> float:
        """Compute the raw metric value."""
        raise NotImplementedError

    def to_score_0_1(self, raw: float) -> float:
        """Map raw metric value to a display score in [0..1] (higher is better)."""
        return _clamp01(raw)


# Registry for all metrics used by the app.
_METRICS: List[Metric] = []


def register_metric(metric: Metric) -> None:
    """Register a metric in the global list."""
    _METRICS.append(metric)


# =============================================================================
# Classic pixel-space metrics
# =============================================================================

class MAE(Metric):
    key = "mae"
    display_name = "MAE (L1)"
    description = "Mean Absolute Error between generated and ground-truth right image (pixel-space)."
    suitability = 4

    def compute_raw(self, left, r_gt, r_gen, meta):
        a = _to_float01(r_gt)
        b = _to_float01(r_gen)
        return float(np.mean(np.abs(a - b)))

    def to_score_0_1(self, raw: float) -> float:
        # Exponential decay: smaller error -> closer to 1.
        return _clamp01(math.exp(-raw * 7.0))


class MSE(Metric):
    key = "mse"
    display_name = "MSE (L2)"
    description = "Mean Squared Error between generated and ground-truth right image (pixel-space)."
    suitability = 3

    def compute_raw(self, left, r_gt, r_gen, meta):
        a = _to_float01(r_gt)
        b = _to_float01(r_gen)
        return float(np.mean((a - b) ** 2))

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01(math.exp(-raw * 70.0))


class RMSEMetric(Metric):
    key = "rmse"
    display_name = "RMSE"
    description = "Root Mean Squared Error between generated and ground-truth right image (pixel-space)."
    suitability = 3

    def compute_raw(self, left, r_gt, r_gen, meta):
        a = _to_float01(r_gt)
        b = _to_float01(r_gen)
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01(math.exp(-raw * 7.0))


class HuberMetric(Metric):
    key = "huber"
    display_name = "Huber (Smooth-L1)"
    description = "Robust error combining L1/L2 between generated and ground-truth right image."
    suitability = 4

    def compute_raw(self, left, r_gt, r_gen, meta):
        # Delta defines the transition point between quadratic and linear error.
        delta = 0.02
        a = _to_float01(r_gt)
        b = _to_float01(r_gen)
        d = a - b
        absd = np.abs(d)
        quad = np.minimum(absd, delta)
        lin = absd - quad

        # Standard Huber formulation (scaled to be smooth at delta).
        loss = 0.5 * quad * quad / delta + lin
        return float(np.mean(loss))

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01(math.exp(-raw * 20.0))


class CharbonnierMetric(Metric):
    key = "charbonnier"
    display_name = "Charbonnier"
    description = "Smooth L1-like robust error (Charbonnier loss) between generated and ground-truth right image."
    suitability = 4

    def compute_raw(self, left, r_gt, r_gen, meta):
        eps = 1e-3
        a = _to_float01(r_gt)
        b = _to_float01(r_gen)
        d = a - b
        return float(np.mean(np.sqrt(d * d + eps * eps)))

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01(math.exp(-raw * 10.0))


class PSNRMetric(Metric):
    key = "psnr"
    display_name = "PSNR (dB)"
    description = (
        "Peak Signal-to-Noise Ratio (dB). Normalization uses a smooth knee from 10 dB → 0 to 40 dB → 1 (clamped)."
    )
    suitability = 3

    def compute_raw(self, left, r_gt, r_gen, meta):
        a = _to_float01(r_gt)
        b = _to_float01(r_gen)
        mse = float(np.mean((a - b) ** 2))

        # Guard against log(0) for identical images.
        if mse <= 1e-12:
            return 100.0

        # Max signal for [0..1] images is 1.
        return float(10.0 * math.log10(1.0 / mse))

    def to_score_0_1(self, raw: float) -> float:
        # Smoothstep with clamping at 40 dB.
        t = (raw - 10.0) / 30.0
        return _smoothstep(t)


# =============================================================================
# SSIM family
# =============================================================================

def _ssim_components(x: np.ndarray, y: np.ndarray, ksize: int = 11, sigma: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SSIM map and contrast-structure (CS) map for a single channel."""

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    mu_x = _gaussian_blur(x, sigma=sigma, ksize=ksize)
    mu_y = _gaussian_blur(y, sigma=sigma, ksize=ksize)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = _gaussian_blur(x * x, sigma=sigma, ksize=ksize) - mu_x2
    sigma_y2 = _gaussian_blur(y * y, sigma=sigma, ksize=ksize) - mu_y2
    sigma_xy = _gaussian_blur(x * y, sigma=sigma, ksize=ksize) - mu_xy

    # Standard SSIM constants for normalized data.
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)

    num_ssim = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den_ssim = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num_ssim / (den_ssim + 1e-12)

    cs_map = (2 * sigma_xy + c2) / (sigma_x2 + sigma_y2 + c2 + 1e-12)
    return ssim_map, cs_map


class SSIMMetric(Metric):
    key = "ssim"
    display_name = "SSIM"
    description = "Structural Similarity Index (mean over RGB channels)."
    suitability = 4

    def compute_raw(self, left, r_gt, r_gen, meta):
        a = _to_float01(r_gt)
        b = _to_float01(r_gen)
        vals = []
        for c in range(3):
            ssim_map, _ = _ssim_components(a[..., c], b[..., c])
            vals.append(_safe_mean(ssim_map))
        return float(np.mean(vals))

    def to_score_0_1(self, raw: float) -> float:
        # SSIM is typically in [-1..1]. Map to [0..1].
        return _clamp01((raw + 1.0) / 2.0)


class MSSSIMMetric(Metric):
    key = "ms_ssim"
    display_name = "MS-SSIM"
    description = "Multi-Scale SSIM (mean over RGB channels)."
    suitability = 5

    def compute_raw(self, left, r_gt, r_gen, meta):
        a = _to_float01(r_gt)
        b = _to_float01(r_gen)
        weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float32)

        def _ms_ssim_single_channel(x: np.ndarray, y: np.ndarray) -> float:
            # Four scales of contrast-structure + final SSIM at the smallest scale.
            mcs = []
            for _ in range(4):
                _, cs = _ssim_components(x, y)
                mcs.append(_safe_mean(cs))
                x = _downsample2(x)
                y = _downsample2(y)

            ssim_map, _ = _ssim_components(x, y)
            mssim = _safe_mean(ssim_map)
            mcs = np.array(mcs, dtype=np.float32)

            out = float(np.prod(np.maximum(mcs, 1e-6) ** weights[:4]) * (max(mssim, 1e-6) ** float(weights[4])))
            return out

        vals = [_ms_ssim_single_channel(a[..., c], b[..., c]) for c in range(3)]
        return float(np.mean(vals))

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01(raw)


class IWSSIMMetric(Metric):
    key = "iw_ssim"
    display_name = "IW-SSIM (approx)"
    description = "Information-weighted SSIM (approx.): SSIM map weighted by local variance."
    suitability = 4

    def compute_raw(self, left, r_gt, r_gen, meta):
        x = _rgb_to_gray(_to_float01(r_gt))
        y = _rgb_to_gray(_to_float01(r_gen))
        ssim_map, _ = _ssim_components(x, y)

        # Approximate "information" by variance.
        mu_x = _gaussian_blur(x, sigma=1.5, ksize=11)
        mu_y = _gaussian_blur(y, sigma=1.5, ksize=11)
        vx = _gaussian_blur(x * x, sigma=1.5, ksize=11) - mu_x * mu_x
        vy = _gaussian_blur(y * y, sigma=1.5, ksize=11) - mu_y * mu_y

        w = np.maximum(vx + vy, 1e-6)
        return float(np.sum(ssim_map * w) / (np.sum(w) + 1e-12))

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01((raw + 1.0) / 2.0)


class UQIMetric(Metric):
    key = "uqi"
    display_name = "UQI"
    description = "Universal Quality Index (global, grayscale)."
    suitability = 3

    def compute_raw(self, left, r_gt, r_gen, meta):
        x = _rgb_to_gray(_to_float01(r_gt)).flatten()
        y = _rgb_to_gray(_to_float01(r_gen)).flatten()

        mx = float(np.mean(x))
        my = float(np.mean(y))
        vx = float(np.var(x))
        vy = float(np.var(y))
        cov = float(np.mean((x - mx) * (y - my)))

        num = 4 * cov * mx * my
        den = (vx + vy) * (mx * mx + my * my) + 1e-12
        return float(num / den)

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01((raw + 1.0) / 2.0)


# =============================================================================
# Gradient / edge metrics
# =============================================================================

def _sobel_mag(gray: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude for a grayscale float image."""

    if _HAS_CV2:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    # Standard 3x3 Sobel kernels.
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    if _HAS_SCIPY:
        gx = ndimage.convolve(gray, kx, mode="reflect")
        gy = ndimage.convolve(gray, ky, mode="reflect")
        return np.sqrt(gx * gx + gy * gy)

    # Slow but dependency-free 2D convolution fallback.
    def conv2(a, k):
        pad = 1
        ap = np.pad(a, ((pad, pad), (pad, pad)), mode="reflect")
        out = np.zeros_like(a, dtype=np.float32)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                out[i, j] = float(np.sum(ap[i : i + 3, j : j + 3] * k))
        return out

    gx = conv2(gray, kx)
    gy = conv2(gray, ky)
    return np.sqrt(gx * gx + gy * gy)


class GradientDiffMetric(Metric):
    key = "grad_diff"
    display_name = "Gradient Difference"
    description = "L1 difference of Sobel gradient magnitudes (grayscale)."
    suitability = 3

    def compute_raw(self, left, r_gt, r_gen, meta):
        x = _rgb_to_gray(_to_float01(r_gt))
        y = _rgb_to_gray(_to_float01(r_gen))
        gx = _sobel_mag(x)
        gy = _sobel_mag(y)
        return float(np.mean(np.abs(gx - gy)))

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01(math.exp(-raw * 3.0))


class GMSDMetric(Metric):
    key = "gmsd"
    display_name = "GMSD"
    description = "Gradient Magnitude Similarity Deviation (grayscale; lower is better)."
    suitability = 4

    def compute_raw(self, left, r_gt, r_gen, meta):
        x = _rgb_to_gray(_to_float01(r_gt))
        y = _rgb_to_gray(_to_float01(r_gen))
        gx = _sobel_mag(x)
        gy = _sobel_mag(y)

        T = 0.0026
        gms = (2 * gx * gy + T) / (gx * gx + gy * gy + T + 1e-12)
        return float(np.std(gms))

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01(math.exp(-raw / 0.15))


# =============================================================================
# Correlation metric
# =============================================================================

class ZNCCMetric(Metric):
    key = "zncc"
    display_name = "ZNCC"
    description = "Zero-mean Normalized Cross Correlation (grayscale, global)."
    suitability = 3

    def compute_raw(self, left, r_gt, r_gen, meta):
        x = _rgb_to_gray(_to_float01(r_gt)).astype(np.float32)
        y = _rgb_to_gray(_to_float01(r_gen)).astype(np.float32)

        x = x - float(np.mean(x))
        y = y - float(np.mean(y))

        num = float(np.mean(x * y))
        den = float(np.sqrt(np.mean(x * x) * np.mean(y * y)) + 1e-12)
        return float(num / den)

    def to_score_0_1(self, raw: float) -> float:
        # ZNCC is typically in [-1..1]. Map to [0..1].
        return _clamp01((raw + 1.0) / 2.0)


# =============================================================================
# Color difference (CIEDE2000)
# =============================================================================

def _srgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (float) to XYZ (D65) in a vectorized way."""

    rgb = rgb.astype(np.float32)

    def inv_gamma(u):
        # Inverse sRGB companding.
        return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)

    r, g, b = inv_gamma(rgb[..., 0]), inv_gamma(rgb[..., 1]), inv_gamma(rgb[..., 2])

    # Matrix for sRGB → XYZ (D65).
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return np.stack([x, y, z], axis=-1)


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ to CIE Lab (D65 reference white)."""

    # D65 reference white.
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn

    delta = 6 / 29

    def f(t):
        return np.where(t > delta**3, np.cbrt(t), (t / (3 * delta**2)) + (4 / 29))

    fx, fy, fz = f(x), f(y), f(z)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def _ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Compute CIEDE2000 color difference in a vectorized form."""

    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    avg_L = 0.5 * (L1 + L2)

    C1 = np.sqrt(a1 * a1 + b1 * b1)
    C2 = np.sqrt(a2 * a2 + b2 * b2)
    avg_C = 0.5 * (C1 + C2)

    G = 0.5 * (1 - np.sqrt((avg_C**7) / (avg_C**7 + 25**7 + 1e-12)))

    a1p = (1 + G) * a1
    a2p = (1 + G) * a2

    C1p = np.sqrt(a1p * a1p + b1 * b1)
    C2p = np.sqrt(a2p * a2p + b2 * b2)
    avg_Cp = 0.5 * (C1p + C2p)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where((C1p * C2p) == 0, 0, dhp)

    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)

    avg_hp = (h1p + h2p) / 2
    avg_hp = np.where(np.abs(h1p - h2p) > 180, avg_hp + 180, avg_hp)
    avg_hp = (avg_hp % 360)
    avg_hp = np.where((C1p * C2p) == 0, h1p + h2p, avg_hp)

    T = (
        1
        - 0.17 * np.cos(np.radians(avg_hp - 30))
        + 0.24 * np.cos(np.radians(2 * avg_hp))
        + 0.32 * np.cos(np.radians(3 * avg_hp + 6))
        - 0.20 * np.cos(np.radians(4 * avg_hp - 63))
    )

    d_ro = 30 * np.exp(-(((avg_hp - 275) / 25) ** 2))
    R_C = 2 * np.sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7 + 1e-12))

    S_L = 1 + (0.015 * ((avg_L - 50) ** 2)) / np.sqrt(20 + ((avg_L - 50) ** 2) + 1e-12)
    S_C = 1 + 0.045 * avg_Cp
    S_H = 1 + 0.015 * avg_Cp * T

    R_T = -np.sin(2 * np.radians(d_ro)) * R_C

    kL = kC = kH = 1.0

    dE = np.sqrt(
        (dLp / (kL * S_L + 1e-12)) ** 2
        + (dCp / (kC * S_C + 1e-12)) ** 2
        + (dHp / (kH * S_H + 1e-12)) ** 2
        + R_T * (dCp / (kC * S_C + 1e-12)) * (dHp / (kH * S_H + 1e-12))
    )

    return dE.astype(np.float32)


class DeltaE2000Metric(Metric):
    key = "deltae2000"
    display_name = "ΔE (CIEDE2000)"
    description = "CIE Lab color difference (lower is better)."
    suitability = 3

    def compute_raw(self, left, r_gt, r_gen, meta):
        a = _to_float01(r_gt)
        b = _to_float01(r_gen)
        lab1 = _xyz_to_lab(_srgb_to_xyz(a))
        lab2 = _xyz_to_lab(_srgb_to_xyz(b))
        dE = _ciede2000(lab1, lab2)
        return float(np.mean(dE))

    def to_score_0_1(self, raw: float) -> float:
        # Empirical mapping: typical ΔE values vary widely; exp(-x/10) is a mild penalty.
        return _clamp01(math.exp(-raw / 10.0))


# =============================================================================
# VIFp (approx)
# =============================================================================

class VIFpApprox(Metric):
    key = "vifp"
    display_name = "VIFp (approx)"
    description = "Visual Information Fidelity (pixel-domain approximation)."
    suitability = 4

    def compute_raw(self, left, r_gt, r_gen, meta):
        ref = _rgb_to_gray(_to_float01(r_gt))
        dist = _rgb_to_gray(_to_float01(r_gen))

        sigma_nsq = 2e-4
        eps = 1e-10

        num = 0.0
        den = 0.0

        # Multi-scale approximation: blur + downsample.
        for _ in range(4):
            ksize = 5
            sigma = 1.2

            mu1 = _gaussian_blur(ref, sigma=sigma, ksize=ksize)
            mu2 = _gaussian_blur(dist, sigma=sigma, ksize=ksize)

            sigma1_sq = _gaussian_blur(ref * ref, sigma=sigma, ksize=ksize) - mu1 * mu1
            sigma2_sq = _gaussian_blur(dist * dist, sigma=sigma, ksize=ksize) - mu2 * mu2
            sigma12 = _gaussian_blur(ref * dist, sigma=sigma, ksize=ksize) - mu1 * mu2

            sigma1_sq = np.maximum(0.0, sigma1_sq)
            sigma2_sq = np.maximum(0.0, sigma2_sq)

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            sv_sq = np.maximum(sv_sq, eps)
            g = np.where(sigma1_sq < eps, 0.0, g)
            sv_sq = np.where(sigma1_sq < eps, sigma2_sq, sv_sq)

            num += float(np.sum(np.log10(1.0 + (g * g) * sigma1_sq / (sv_sq + sigma_nsq))))
            den += float(np.sum(np.log10(1.0 + sigma1_sq / sigma_nsq)))

            ref = _downsample2(ref)
            dist = _downsample2(dist)

        if den <= 0:
            return 0.0

        # VIFp is often interpreted in [0..1] (approx).
        return float(np.clip(num / den, 0.0, 1.0))

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01(raw)


# =============================================================================
# FSIM (approx)
# =============================================================================

class FSIMApprox(Metric):
    key = "fsim"
    display_name = "FSIM (approx)"
    description = "Feature Similarity (approx.): gradient similarity weighted by gradient strength."
    suitability = 4

    def compute_raw(self, left, r_gt, r_gen, meta):
        x = _rgb_to_gray(_to_float01(r_gt))
        y = _rgb_to_gray(_to_float01(r_gen))
        gx = _sobel_mag(x)
        gy = _sobel_mag(y)

        T = 0.01
        s = (2 * gx * gy + T) / (gx * gx + gy * gy + T + 1e-12)
        w = np.maximum(gx, gy)
        return float(np.sum(s * w) / (np.sum(w) + 1e-12))

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01(raw)


# =============================================================================
# CW-SSIM (approx; requires cv2)
# =============================================================================

class CWSSIMApprox(Metric):
    key = "cw_ssim"
    display_name = "CW-SSIM (approx)"
    description = "Complex Wavelet SSIM (approx.): complex Gabor responses; requires OpenCV."
    suitability = 4

    def is_available(self) -> Tuple[bool, str]:
        if not _HAS_CV2:
            return False, "Missing dependency: opencv-python (cv2)"
        return True, ""

    def compute_raw(self, left, r_gt, r_gen, meta):
        x = _rgb_to_gray(_to_float01(r_gt))
        y = _rgb_to_gray(_to_float01(r_gen))

        # Single Gabor kernel as a rough proxy for complex wavelet decomposition.
        ksize = 21
        sigma = 4.0
        theta = 0.0
        lambd = 10.0
        gamma = 0.5

        real = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F)
        imag = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=np.pi / 2, ktype=cv2.CV_32F)

        xr = cv2.filter2D(x, cv2.CV_32F, real, borderType=cv2.BORDER_REFLECT)
        xi = cv2.filter2D(x, cv2.CV_32F, imag, borderType=cv2.BORDER_REFLECT)
        yr = cv2.filter2D(y, cv2.CV_32F, real, borderType=cv2.BORDER_REFLECT)
        yi = cv2.filter2D(y, cv2.CV_32F, imag, borderType=cv2.BORDER_REFLECT)

        x_complex = xr + 1j * xi
        y_complex = yr + 1j * yi

        # Similarity of complex responses (global aggregation).
        K = 1e-6
        num = 2.0 * np.abs(np.sum(np.conj(x_complex) * y_complex)) + K
        den = np.sum(np.abs(x_complex) ** 2) + np.sum(np.abs(y_complex) ** 2) + K
        return float(num / (den + 1e-12))

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01(raw)


# =============================================================================
# PSNR-HVS (approx; DCT weighting)
# =============================================================================

class PSNRHVSApprox(Metric):
    key = "psnr_hvs"
    display_name = "PSNR-HVS (approx, dB)"
    description = (
        "PSNR-HVS approximation: block-DCT weighted MSE. Normalization uses smooth knee 10 dB → 0 to 40 dB → 1."
    )
    suitability = 3

    def is_available(self) -> Tuple[bool, str]:
        if not (_HAS_CV2 or _HAS_SCIPY):
            return False, "Missing dependency: cv2 or scipy (needed for DCT)"
        return True, ""

    def compute_raw(self, left, r_gt, r_gen, meta):
        x = _rgb_to_gray(_to_float01(r_gt))
        y = _rgb_to_gray(_to_float01(r_gen))

        # JPEG quantization table used here as a crude HVS weighting proxy.
        Q = np.array(
            [
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
            ],
            dtype=np.float32,
        )

        W = 1.0 / (Q.astype(np.float32) + 1e-6)
        W = W / np.max(W)

        h, w = x.shape

        # Pad to full 8x8 blocks.
        H = int(math.ceil(h / 8) * 8)
        Ww = int(math.ceil(w / 8) * 8)
        xpad = np.pad(x, ((0, H - h), (0, Ww - w)), mode="reflect")
        ypad = np.pad(y, ((0, H - h), (0, Ww - w)), mode="reflect")

        def dct2(block: np.ndarray) -> np.ndarray:
            # Prefer OpenCV DCT; otherwise use SciPy DCT (orthonormal).
            if _HAS_CV2:
                return cv2.dct(block.astype(np.float32))
            b = scipy_dct(block.astype(np.float32), axis=0, norm="ortho")
            b = scipy_dct(b, axis=1, norm="ortho")
            return b

        se = 0.0
        count = 0
        for i in range(0, H, 8):
            for j in range(0, Ww, 8):
                bx = xpad[i : i + 8, j : j + 8]
                by = ypad[i : i + 8, j : j + 8]
                dx = dct2(bx)
                dy = dct2(by)
                diff = (dx - dy) * W
                se += float(np.mean(diff * diff))
                count += 1

        mse_w = se / max(1, count)
        if mse_w <= 1e-12:
            return 100.0

        return float(10.0 * math.log10(1.0 / mse_w))

    def to_score_0_1(self, raw: float) -> float:
        t = (raw - 10.0) / 30.0
        return _smoothstep(t)


# =============================================================================
# Feature matching metric (ORB) – requires cv2
# =============================================================================

class ORBMatchMetric(Metric):
    key = "orb_match"
    display_name = "Keypoint Match (ORB)"
    description = "Feature-matching quality between generated and ground-truth right images (ORB inlier ratio)."
    suitability = 3

    def is_available(self) -> Tuple[bool, str]:
        if not _HAS_CV2:
            return False, "Missing dependency: opencv-python (cv2)"
        return True, ""

    def compute_raw(self, left, r_gt, r_gen, meta):
        img1 = cv2.cvtColor(r_gt, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(r_gen, cv2.COLOR_RGB2GRAY)

        # Detect and describe keypoints.
        orb = cv2.ORB_create(nfeatures=2000)
        k1, d1 = orb.detectAndCompute(img1, None)
        k2, d2 = orb.detectAndCompute(img2, None)
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            return 0.0

        # Cross-check matching for simplicity.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d1, d2)
        if not matches:
            return 0.0

        # Keep only strongest matches.
        matches = sorted(matches, key=lambda m: m.distance)[: min(200, len(matches))]
        pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([k2[m.trainIdx].pt for m in matches])

        # Estimate geometric consistency via RANSAC homography.
        _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
        if mask is None:
            return 0.0

        inliers = int(mask.ravel().sum())
        return float(np.clip(inliers / max(1, len(matches)), 0.0, 1.0))

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01(raw)


# =============================================================================
# Proxy downstream metric (edge F1)
# =============================================================================

class EdgeF1Metric(Metric):
    key = "edge_f1"
    display_name = "Task Proxy: Edge F1"
    description = "Proxy task: F1 score between edge maps (Canny or gradient-threshold fallback)."
    suitability = 4

    def compute_raw(self, left, r_gt, r_gen, meta):
        gt = _rgb_to_gray(_to_float01(r_gt))
        ge = _rgb_to_gray(_to_float01(r_gen))

        # Compute binary edge maps.
        if _HAS_CV2:
            gt_u8 = (gt * 255).astype(np.uint8)
            ge_u8 = (ge * 255).astype(np.uint8)
            e1 = cv2.Canny(gt_u8, 80, 160)
            e2 = cv2.Canny(ge_u8, 80, 160)
            e1 = e1 > 0
            e2 = e2 > 0
        else:
            m1 = _sobel_mag(gt)
            m2 = _sobel_mag(ge)
            t1 = float(np.percentile(m1, 90))
            t2 = float(np.percentile(m2, 90))
            e1 = m1 >= t1
            e2 = m2 >= t2

        # Confusion matrix.
        tp = int(np.logical_and(e1, e2).sum())
        fp = int(np.logical_and(~e1, e2).sum())
        fn = int(np.logical_and(e1, ~e2).sum())

        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        if precision + recall == 0:
            return 0.0

        return float(np.clip(2 * precision * recall / (precision + recall), 0.0, 1.0))

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01(raw)


# =============================================================================
# Learned metrics (Torch) – wrappers
# =============================================================================
#
# These metrics typically require GPU/CPU-heavy inference. We keep model instances
# cached per metric instance. The global torch lock serializes access to prevent
# thrashing in multi-threaded batch mode.
# =============================================================================

class LPIPSMetric(Metric):
    key = "lpips"
    display_name = "LPIPS"
    description = "Learned Perceptual Image Patch Similarity (requires torch + lpips; lower distance → higher score)."
    suitability = 5

    def __init__(self) -> None:
        self._net = None
        self._device = None

    def is_available(self) -> Tuple[bool, str]:
        if not _HAS_TORCH:
            return False, "Missing dependency: torch"
        try:
            import lpips  # type: ignore

            _ = lpips
        except Exception:
            return False, "Missing dependency: lpips"
        return True, ""

    def _get_net(self):
        import lpips  # type: ignore

        # Lazily load the model on first use.
        if self._net is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._net = lpips.LPIPS(net="alex").to(self._device).eval()
        return self._net, self._device

    def compute_raw(self, left, r_gt, r_gen, meta):
        with QtCore.QMutexLocker(_TORCH_LOCK):
            net, device = self._get_net()
            with torch.no_grad():
                a = torch.from_numpy(_to_float01(r_gt).transpose(2, 0, 1)).unsqueeze(0).to(device)
                b = torch.from_numpy(_to_float01(r_gen).transpose(2, 0, 1)).unsqueeze(0).to(device)

                # LPIPS expects inputs in [-1..1].
                a = a * 2 - 1
                b = b * 2 - 1

                d = net(a, b)
                return float(d.item())

    def to_score_0_1(self, raw: float) -> float:
        # Typical LPIPS is ~[0..1+] where lower is better.
        return _clamp01(math.exp(-raw * 2.5))


class VGGPerceptualMetric(Metric):
    key = "vgg_perc"
    display_name = "Perceptual (VGG16)"
    description = "VGG16 feature L2 distance (requires torch+torchvision). Lower distance → higher score."
    suitability = 4

    def __init__(self) -> None:
        self._device = None
        self._vgg = None

        # Indices in VGG feature extractor where features are collected.
        self._layers = [3, 8, 15, 22]

    def is_available(self) -> Tuple[bool, str]:
        if not (_HAS_TORCH and _HAS_TORCHVISION):
            return False, "Missing dependency: torch and/or torchvision"
        return True, ""

    def _get_model(self):
        # Lazily load VGG16 features.
        if self._vgg is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features
            except Exception:
                # Backward-compatible fallback.
                vgg = torchvision.models.vgg16(pretrained=True).features
            self._vgg = vgg.to(self._device).eval()
        return self._vgg, self._device

    def _preprocess(self, img_u8: np.ndarray, device: str) -> Any:
        # Convert to tensor and apply ImageNet normalization.
        x = _to_float01(img_u8)
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        return (t - mean) / std

    def compute_raw(self, left, r_gt, r_gen, meta):
        with QtCore.QMutexLocker(_TORCH_LOCK):
            vgg, device = self._get_model()
            a = self._preprocess(r_gt, device)
            b = self._preprocess(r_gen, device)

            with torch.no_grad():
                feats_a = []
                feats_b = []
                x1, x2 = a, b

                # Forward pass through VGG feature extractor.
                for i, layer in enumerate(vgg):
                    x1 = layer(x1)
                    x2 = layer(x2)
                    if i in self._layers:
                        feats_a.append(x1)
                        feats_b.append(x2)

                # Aggregate L2 distances across selected feature maps.
                dist = 0.0
                for fa, fb in zip(feats_a, feats_b):
                    dist += float(torch.mean((fa - fb) ** 2).item())

            return float(dist)

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01(math.exp(-raw * 15.0))


class CLIPSimMetric(Metric):
    key = "clip_sim"
    display_name = "CLIP Similarity"
    description = "Cosine similarity in CLIP embedding space (requires open_clip_torch OR transformers)."
    suitability = 3

    def __init__(self) -> None:
        self._device = None
        self._mode = None
        self._model = None
        self._processor = None

    def is_available(self) -> Tuple[bool, str]:
        if not _HAS_TORCH:
            return False, "Missing dependency: torch"

        # Try OpenCLIP first, then HF Transformers.
        try:
            import open_clip  # type: ignore

            _ = open_clip
            return True, ""
        except Exception:
            try:
                import transformers  # type: ignore

                _ = transformers
                return True, ""
            except Exception:
                return False, "Missing dependency: open_clip_torch or transformers"

    def _ensure_model(self):
        if self._model is not None:
            return

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            import open_clip  # type: ignore

            model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
            self._model = model.to(self._device).eval()
            self._mode = "open_clip"
        except Exception:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore

            self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self._device).eval()
            self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._mode = "hf"

    def compute_raw(self, left, r_gt, r_gen, meta):
        with QtCore.QMutexLocker(_TORCH_LOCK):
            self._ensure_model()
            device = self._device

            if self._mode == "open_clip":
                # Minimal tensor preprocess (avoids PIL dependency).
                def emb(img_u8: np.ndarray) -> Any:
                    x = torch.from_numpy(_to_float01(img_u8).transpose(2, 0, 1)).unsqueeze(0).to(device)
                    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
                    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
                    x = (x - mean) / std
                    with torch.no_grad():
                        e = self._model.encode_image(x)  # type: ignore
                        e = e / (e.norm(dim=-1, keepdim=True) + 1e-12)
                    return e

                e1 = emb(r_gt)
                e2 = emb(r_gen)
                return float((e1 * e2).sum().item())

            # Transformers path.
            from PIL import Image  # type: ignore

            a = Image.fromarray(r_gt)
            b = Image.fromarray(r_gen)
            inputs = self._processor(images=[a, b], return_tensors="pt")  # type: ignore
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self._model.get_image_features(**inputs)  # type: ignore
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
            return float((feats[0] * feats[1]).sum().item())

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01((raw + 1.0) / 2.0)


class DISTSMetric(Metric):
    key = "dists"
    display_name = "DISTS"
    description = "Deep Image Structure and Texture Similarity (requires torch + piq)."
    suitability = 5

    def __init__(self) -> None:
        self._metric = None
        self._device = None

    def is_available(self) -> Tuple[bool, str]:
        if not _HAS_TORCH:
            return False, "Missing dependency: torch"
        try:
            import piq  # type: ignore

            _ = piq
        except Exception:
            return False, "Missing dependency: piq"
        return True, ""

    def _ensure(self):
        if self._metric is not None:
            return
        import piq  # type: ignore

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._metric = piq.DISTS().to(self._device).eval()  # type: ignore

    def compute_raw(self, left, r_gt, r_gen, meta):
        with QtCore.QMutexLocker(_TORCH_LOCK):
            self._ensure()
            device = self._device
            with torch.no_grad():
                a = torch.from_numpy(_to_float01(r_gt).transpose(2, 0, 1)).unsqueeze(0).to(device)
                b = torch.from_numpy(_to_float01(r_gen).transpose(2, 0, 1)).unsqueeze(0).to(device)

                # piq.DISTS returns a distance (lower is better).
                d = self._metric(a, b)  # type: ignore
                return float(d.item())

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01(math.exp(-raw * 2.0))


class PieAPPMetric(Metric):
    key = "pieapp"
    display_name = "PieAPP (if available)"
    description = "Perceptual image-error metric (requires torch + piq with PieAPP support)."
    suitability = 4

    def __init__(self) -> None:
        self._metric = None
        self._device = None

    def is_available(self) -> Tuple[bool, str]:
        if not _HAS_TORCH:
            return False, "Missing dependency: torch"
        try:
            import piq  # type: ignore

            if not hasattr(piq, "PieAPP"):
                return False, "piq is installed, but PieAPP is not available in this build"
        except Exception:
            return False, "Missing dependency: piq"
        return True, ""

    def _ensure(self):
        if self._metric is not None:
            return
        import piq  # type: ignore

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._metric = piq.PieAPP().to(self._device).eval()  # type: ignore

    def compute_raw(self, left, r_gt, r_gen, meta):
        with QtCore.QMutexLocker(_TORCH_LOCK):
            self._ensure()
            device = self._device
            with torch.no_grad():
                a = torch.from_numpy(_to_float01(r_gt).transpose(2, 0, 1)).unsqueeze(0).to(device)
                b = torch.from_numpy(_to_float01(r_gen).transpose(2, 0, 1)).unsqueeze(0).to(device)
                d = self._metric(a, b)  # type: ignore
                return float(d.item())

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01(math.exp(-raw * 1.5))


class DINOSimMetric(Metric):
    key = "dino_sim"
    display_name = "DINO Similarity"
    description = "Self-supervised ViT embedding cosine similarity (requires torch + timm OR transformers)."
    suitability = 3

    def __init__(self) -> None:
        self._device = None
        self._mode = None
        self._model = None
        self._processor = None

    def is_available(self) -> Tuple[bool, str]:
        if not _HAS_TORCH:
            return False, "Missing dependency: torch"
        try:
            import timm  # type: ignore

            _ = timm
            return True, ""
        except Exception:
            try:
                import transformers  # type: ignore

                _ = transformers
                return True, ""
            except Exception:
                return False, "Missing dependency: timm or transformers"

    def _ensure(self):
        if self._model is not None:
            return

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Prefer timm DINO; fallback to HF DINOv2.
        try:
            import timm  # type: ignore

            self._model = timm.create_model("vit_small_patch16_224_dino", pretrained=True, num_classes=0)
            self._model = self._model.to(self._device).eval()
            self._mode = "timm"
        except Exception:
            from transformers import AutoImageProcessor, AutoModel  # type: ignore

            self._processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self._model = AutoModel.from_pretrained("facebook/dinov2-base").to(self._device).eval()
            self._mode = "hf"

    def compute_raw(self, left, r_gt, r_gen, meta):
        with QtCore.QMutexLocker(_TORCH_LOCK):
            self._ensure()
            device = self._device

            if self._mode == "timm":
                def emb(img_u8: np.ndarray) -> Any:
                    x = torch.from_numpy(_to_float01(img_u8).transpose(2, 0, 1)).unsqueeze(0).to(device)
                    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                    x = (x - mean) / std
                    with torch.no_grad():
                        e = self._model(x)  # type: ignore
                        e = e / (e.norm(dim=-1, keepdim=True) + 1e-12)
                    return e

                e1 = emb(r_gt)
                e2 = emb(r_gen)
                return float((e1 * e2).sum().item())

            from PIL import Image  # type: ignore

            a = Image.fromarray(r_gt)
            b = Image.fromarray(r_gen)
            inputs = self._processor(images=[a, b], return_tensors="pt")  # type: ignore
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model(**inputs)  # type: ignore

                # Use CLS token embedding as a global image descriptor.
                feats = out.last_hidden_state[:, 0, :]
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)

            return float((feats[0] * feats[1]).sum().item())

    def to_score_0_1(self, raw: float) -> float:
        return _clamp01((raw + 1.0) / 2.0)


# =============================================================================
# Stereo consistency metric (extra; uses left)
# =============================================================================

class StereoDispConsistencyEPE(Metric):
    key = "stereo_disp_epe"
    display_name = "Stereo Extra: Disp Consistency (EPE)"
    description = "EPE between disparities from (L, R_gt) vs (L, R_gen) via StereoSGBM (OpenCV)."
    suitability = 4
    is_stereo_extra = True

    def is_available(self) -> Tuple[bool, str]:
        if not _HAS_CV2:
            return False, "Missing dependency: opencv-python (cv2)"
        return True, ""

    def compute_raw(self, left, r_gt, r_gen, meta):
        # Convert to grayscale for stereo block matching.
        L = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        Rgt = cv2.cvtColor(r_gt, cv2.COLOR_RGB2GRAY)
        Rgen = cv2.cvtColor(r_gen, cv2.COLOR_RGB2GRAY)

        # Conservative SGBM parameters suitable as a quick consistency signal.
        min_disp = 0
        num_disp = 16 * 8
        block = 5

        sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block,
            P1=8 * 1 * block * block,
            P2=32 * 1 * block * block,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=1,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        d_gt = sgbm.compute(L, Rgt).astype(np.float32) / 16.0
        d_gen = sgbm.compute(L, Rgen).astype(np.float32) / 16.0

        # Evaluate only valid disparity pixels.
        mask = (d_gt > 0) & (d_gen > 0) & np.isfinite(d_gt) & np.isfinite(d_gen)
        if int(mask.sum()) < 100:
            return float("inf")

        return float(np.mean(np.abs(d_gt[mask] - d_gen[mask])))

    def to_score_0_1(self, raw: float) -> float:
        if not np.isfinite(raw):
            return 0.0
        return _clamp01(math.exp(-raw / 5.0))


# =============================================================================
# Register metrics
# =============================================================================

register_metric(MAE())
register_metric(MSE())
register_metric(RMSEMetric())
register_metric(HuberMetric())
register_metric(CharbonnierMetric())
register_metric(PSNRMetric())
register_metric(PSNRHVSApprox())
register_metric(SSIMMetric())
register_metric(MSSSIMMetric())
register_metric(IWSSIMMetric())
register_metric(UQIMetric())
register_metric(VIFpApprox())
register_metric(FSIMApprox())
register_metric(GMSDMetric())
register_metric(GradientDiffMetric())
register_metric(ZNCCMetric())
register_metric(DeltaE2000Metric())
register_metric(CWSSIMApprox())
register_metric(ORBMatchMetric())
register_metric(EdgeF1Metric())
register_metric(LPIPSMetric())
register_metric(DISTSMetric())
register_metric(PieAPPMetric())
register_metric(VGGPerceptualMetric())
register_metric(CLIPSimMetric())
register_metric(DINOSimMetric())
register_metric(StereoDispConsistencyEPE())


# =============================================================================
# Folder scanning
# =============================================================================

def find_collections(root: str) -> List[CollectionItem]:
    """Recursively scan for collection folders.

    A folder is considered a collection if it contains all required files:
      left.jpg, right.jpg, generated.jpg, meta.json

    When a collection is found, scanning does not descend into its subfolders.
    """

    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return []

    required = {"left.jpg", "right.jpg", "generated.jpg", "meta.json"}
    out: List[CollectionItem] = []

    for dirpath, dirnames, filenames in os.walk(root):
        files = set(filenames)
        if required.issubset(files):
            name = os.path.basename(dirpath)
            out.append(
                CollectionItem(
                    name=name,
                    path=dirpath,
                    left_path=os.path.join(dirpath, "left.jpg"),
                    right_path=os.path.join(dirpath, "right.jpg"),
                    gen_path=os.path.join(dirpath, "generated.jpg"),
                    meta_path=os.path.join(dirpath, "meta.json"),
                )
            )

            # Stop recursion inside a valid collection folder.
            dirnames[:] = []

    out.sort(key=lambda x: x.name.lower())
    return out


def _load_meta(meta_path: str) -> Dict[str, Any]:
    """Load metadata JSON from meta.json; returns {} on failure."""

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# =============================================================================
# Batch Metric Engine
# =============================================================================

class ComputeWorker(QtCore.QObject, QtCore.QRunnable):
    """Worker that computes ALL registered metrics for a single collection.

    Runs inside Qt's global thread pool.
    """

    finished = QtCore.Signal(str, list, int)  # collection_path, results, run_id
    failed = QtCore.Signal(str, str, int)  # collection_path, message, run_id

    def __init__(self, item: CollectionItem, run_id: int):
        QtCore.QObject.__init__(self)
        QtCore.QRunnable.__init__(self)
        self.item = item
        self.run_id = run_id
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            # Load images and metadata.
            left = _read_image_rgb(self.item.left_path)
            right_gt = _read_image_rgb(self.item.right_path)
            right_gen = _read_image_rgb(self.item.gen_path)
            meta = _load_meta(self.item.meta_path)

            # Align sizes by center-cropping to the smallest common dimensions.
            # This protects metrics from mismatched resolutions.
            h = min(left.shape[0], right_gt.shape[0], right_gen.shape[0])
            w = min(left.shape[1], right_gt.shape[1], right_gen.shape[1])

            def center_crop(img: np.ndarray) -> np.ndarray:
                hh, ww = img.shape[:2]
                y0 = max(0, (hh - h) // 2)
                x0 = max(0, (ww - w) // 2)
                return img[y0 : y0 + h, x0 : x0 + w]

            left = center_crop(left)
            right_gt = center_crop(right_gt)
            right_gen = center_crop(right_gen)

            results: List[MetricResult] = []

            # Compute metric-by-metric; failure of one metric must not abort the entire collection.
            for m in _METRICS:
                ok, reason = m.is_available()
                if not ok:
                    results.append(
                        MetricResult(
                            key=m.key,
                            display_name=m.display_name,
                            description=m.description,
                            raw_value=None,
                            score_0_1=None,
                            available=False,
                            details=reason,
                        )
                    )
                    continue

                try:
                    raw = m.compute_raw(left, right_gt, right_gen, meta)
                    score = m.to_score_0_1(raw)

                    # Human-readable notes for common units.
                    details = ""
                    if m.key in {"psnr", "psnr_hvs"}:
                        details = "dB (higher is better; score uses 10→0 .. 40→1 smooth knee)"
                    if m.key in {"mae", "mse", "rmse", "huber", "charbonnier"}:
                        details = "error (lower is better)"
                    if m.key in {"deltae2000"}:
                        details = "ΔE (lower is better)"
                    if m.key == "stereo_disp_epe":
                        details = "px EPE (lower is better)"

                    results.append(
                        MetricResult(
                            key=m.key,
                            display_name=m.display_name,
                            description=m.description,
                            raw_value=float(raw) if np.isfinite(raw) else float("inf"),
                            score_0_1=float(np.clip(score, 0.0, 1.0)),
                            available=True,
                            details=details,
                        )
                    )
                except Exception as e:
                    # Mark metric as unavailable for this run, but keep the batch alive.
                    results.append(
                        MetricResult(
                            key=m.key,
                            display_name=m.display_name,
                            description=m.description,
                            raw_value=None,
                            score_0_1=None,
                            available=False,
                            details=f"Error: {e}",
                        )
                    )

            # Stable sorting for UI table.
            results.sort(key=lambda r: r.display_name.lower())
            self.finished.emit(self.item.path, results, self.run_id)

        except Exception as e:
            msg = f"{e}\n\n{traceback.format_exc()}"
            self.failed.emit(self.item.path, msg, self.run_id)


class MetricEngine(QtCore.QObject):
    """Coordinates a batch run across collections and stores results in a cache."""

    progress = QtCore.Signal(int, int)  # done, total
    collection_done = QtCore.Signal(str)  # collection_path
    batch_finished = QtCore.Signal()
    error = QtCore.Signal(str, str)  # collection_path, message

    def __init__(self) -> None:
        super().__init__()

        # Cache: collection_path -> metric_key -> MetricResult
        self._cache: Dict[str, Dict[str, MetricResult]] = {}

        self._lock = QtCore.QMutex()
        self._active_workers: List[ComputeWorker] = []

        # Run token to ignore stale worker callbacks from previous runs.
        self._run_id: int = 0

        self._done = 0
        self._total = 0

    def clear_cache(self) -> None:
        """Clear all cached results."""
        with QtCore.QMutexLocker(self._lock):
            self._cache.clear()

    def get_cached(self, collection_path: str) -> Optional[Dict[str, MetricResult]]:
        """Fetch cached results for a collection, if available."""
        with QtCore.QMutexLocker(self._lock):
            return self._cache.get(collection_path)

    def run_all(self, items: List[CollectionItem]) -> None:
        """Run batch computation for all collections."""

        # New run token.
        # NOTE: Qt Signal(int) is a 32-bit signed C int in PySide6.
        # Keep run_id within range to avoid shiboken overflow.
        self._run_id = int(time.monotonic_ns() & 0x7FFFFFFF)

        self._done = 0
        self._total = len(items)
        self.clear_cache()

        if self._total == 0:
            self.progress.emit(0, 0)
            self.batch_finished.emit()
            return

        pool = QtCore.QThreadPool.globalInstance()

        # Keep some parallelism, but avoid runaway contention (especially for torch models).
        pool.setMaxThreadCount(max(1, min(4, os.cpu_count() or 4)))

        self.progress.emit(self._done, self._total)

        self._active_workers.clear()
        for it in items:
            w = ComputeWorker(it, self._run_id)
            w.finished.connect(self._on_worker_finished)
            w.failed.connect(self._on_worker_failed)
            self._active_workers.append(w)
            pool.start(w)

    @QtCore.Slot(str, list, int)
    def _on_worker_finished(self, collection_path: str, results: List[MetricResult], run_id: int) -> None:
        # Ignore stale callbacks from older runs.
        if run_id != self._run_id:
            return

        with QtCore.QMutexLocker(self._lock):
            self._cache[collection_path] = {r.key: r for r in results}

        self._done += 1
        self.collection_done.emit(collection_path)
        self.progress.emit(self._done, self._total)

        if self._done >= self._total:
            self._active_workers.clear()
            self.batch_finished.emit()

    @QtCore.Slot(str, str, int)
    def _on_worker_failed(self, collection_path: str, message: str, run_id: int) -> None:
        if run_id != self._run_id:
            return

        self._done += 1
        self.error.emit(collection_path, message)
        self.progress.emit(self._done, self._total)

        if self._done >= self._total:
            self._active_workers.clear()
            self.batch_finished.emit()


# =============================================================================
# UI components
# =============================================================================

def apply_modern_dark_palette(app: QtWidgets.QApplication) -> None:
    """Apply a modern dark theme via Fusion palette + centralized stylesheet.

    NOTE:
    The UI intentionally avoids QTabWidget-based chrome for the main view and
    image switching. Instead it uses segmented QToolButtons + QStackedWidget to
    prevent common Fusion/QSS artefacts (double borders, stray rounded corners,
    tab-base lines) across platforms.
    """

    app.setStyle("Fusion")
    palette = QtGui.QPalette()

    bg = QtGui.QColor(22, 24, 28)
    card = QtGui.QColor(18, 20, 24)
    panel = QtGui.QColor(30, 33, 38)
    text = QtGui.QColor(230, 233, 240)
    disabled = QtGui.QColor(140, 145, 155)
    accent = QtGui.QColor(87, 157, 255)

    palette.setColor(QtGui.QPalette.Window, bg)
    palette.setColor(QtGui.QPalette.WindowText, text)
    palette.setColor(QtGui.QPalette.Base, card)
    palette.setColor(QtGui.QPalette.AlternateBase, panel)
    palette.setColor(QtGui.QPalette.ToolTipBase, panel)
    palette.setColor(QtGui.QPalette.ToolTipText, text)
    palette.setColor(QtGui.QPalette.Text, text)
    palette.setColor(QtGui.QPalette.Button, panel)
    palette.setColor(QtGui.QPalette.ButtonText, text)
    palette.setColor(QtGui.QPalette.Highlight, accent)
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(10, 10, 10))

    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, disabled)
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, disabled)
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, disabled)

    app.setPalette(palette)

    qss = """
    QMainWindow { background: #16181C; }
    QWidget { color: #E6E9F0; }

    /* Inputs */
    QLineEdit, QComboBox, QSpinBox, QTextEdit {
        background: #121418;
        border: 1px solid #2C313A;
        border-radius: 10px;
        padding: 8px 10px;
        selection-background-color: #579DFF;
    }

    /* meta viewer inside the overview card: no extra border box */
    QTextEdit#MetaView {
        border: none;
        background: transparent;
        padding: 0px;
    }

    /* Primary buttons */
    QPushButton {
        background: #1E2126;
        border: 1px solid #2C313A;
        border-radius: 10px;
        padding: 8px 12px;
    }
    QPushButton:hover { border: 1px solid #3D4350; }
    QPushButton:pressed { background: #171A1F; }

    /* Segmented navigation (used instead of QTabWidget) */
    QToolButton[segmented="true"] {
        background: #1E2126;
        border: 1px solid #2C313A;
        border-radius: 10px;
        padding: 8px 12px;
    }
    QToolButton[segmented="true"]:hover { border: 1px solid #3D4350; }
    QToolButton[segmented="true"]:checked {
        background: #121418;
        border: 1px solid #579DFF;
    }

    /* Cards */
    QFrame#Card {
        background: #121418;
        border: 1px solid #2C313A;
        border-radius: 12px;
    }

    /* Lists */
    QListWidget {
        background: #121418;
        border: 1px solid #2C313A;
        border-radius: 12px;
        padding: 6px;
    }
    QListWidget::item { padding: 8px; border-radius: 8px; }
    QListWidget::item:selected { background: #2B4E7F; }

    /* Collections list lives inside a Card: remove its own border to avoid double frames / scrollbar artefacts */
    QListWidget#CollectionsList {
        background: transparent;
        border: none;
        border-radius: 0px;
        padding: 0px;
    }
    QListWidget#CollectionsList::item { padding: 8px; border-radius: 8px; }
    QListWidget#CollectionsList::item:selected { background: #2B4E7F; }


    /* Tables */
    QTableWidget {
        background: #121418;
        border: 1px solid #2C313A;
        border-radius: 12px;
        gridline-color: #2C313A;
    }
    QHeaderView::section {
        background: #1E2126;
        padding: 8px;
        border: none;
        border-right: 1px solid #2C313A;
        border-bottom: 1px solid #2C313A;
    }

    
    /* Metrics table in its own rounded "card" so the header corners are properly rounded */
    QFrame#TableCard {
        background: #121418;
        border: 1px solid #2C313A;
        border-radius: 12px;
    }

    QTableWidget#MetricsTable {
        background: transparent;
        border: none;
        border-radius: 0px;
        gridline-color: #2C313A;
    }
    QTableWidget#MetricsTable::item { padding: 6px; }

    QTableWidget#MetricsTable QHeaderView { background: transparent; }
    QTableWidget#MetricsTable QHeaderView::section {
        background: #1E2126;
        padding: 8px;
        border: none;
        border-right: 1px solid #2C313A;
        border-bottom: 1px solid #2C313A;
    }
    QTableWidget#MetricsTable QHeaderView::section:first {
        border-top-left-radius: 12px;
    }
    QTableWidget#MetricsTable QHeaderView::section:last {
        border-top-right-radius: 12px;
        border-right: none;
    }
    QTableWidget#MetricsTable QTableCornerButton::section {
        background: #1E2126;
        border: none;
        border-right: 1px solid #2C313A;
        border-bottom: 1px solid #2C313A;
        border-top-left-radius: 12px;
    }
/* Progress */
    QProgressBar {
        border: 1px solid #2C313A;
        border-radius: 10px;
        text-align: center;
        background: #121418;
        height: 18px;
    }
    QProgressBar::chunk { background-color: #579DFF; border-radius: 10px; }

    /* Splitters
       - Main splitter handle is invisible (still draggable)
       - Overview splitter gets a crisp 1px divider
    */
    QSplitter::handle { background: transparent; }
    QSplitter#MainSplit::handle:horizontal { width: 8px; }
    QSplitter#OverviewSplit::handle { background: #2C313A; }
    QSplitter#OverviewSplit::handle:horizontal { width: 1px; }
    QSplitter#OverviewSplit::handle:vertical { height: 1px; }

    /* Scroll areas: let the surrounding Card provide the border */
    QScrollArea { background: transparent; border: none; }
    QScrollArea::viewport { background: transparent; }
    QScrollArea::corner { background: transparent; }

    QLabel#ImageFrame {
        background: transparent;
        border: none;
        border-radius: 0px;
        padding: 0px;
    }

    /* Overview root card: behave like a normal card (rounded corners on all sides) */
    QFrame#OverviewCard {
        background: #121418;
        border: 1px solid #2C313A;
        border-radius: 12px;
    }

    """
    app.setStyleSheet(qss)


class ImagePanel(QtWidgets.QWidget):
    """A titled image display panel using a QLabel for the pixmap."""

    def __init__(self, title: str, min_w: int = 320, min_h: int = 240) -> None:
        super().__init__()

        self.title = QtWidgets.QLabel(title)
        self.title.setStyleSheet("font-weight: 600;")

        self.img = QtWidgets.QLabel()
        self.img.setObjectName("ImageFrame")
        self.img.setMinimumSize(min_w, min_h)
        self.img.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Do not use scaledContents; we scale manually for quality.
        self.img.setScaledContents(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.title)
        layout.addWidget(self.img, 1)

        self._path: Optional[str] = None

    def set_image(self, path: str) -> None:
        """Set image path and rerender."""
        self._path = path
        self._render()

    def _render(self) -> None:
        """Scale and display the current image to fit the panel."""
        if not self._path or not os.path.isfile(self._path):
            self.img.setPixmap(QtGui.QPixmap())
            self.img.setText("(missing)")
            return

        pix = QtGui.QPixmap(self._path)
        if pix.isNull():
            self.img.setPixmap(QtGui.QPixmap())
            self.img.setText("(unreadable)")
            return

        # Leave a small padding inside the frame.
        target = self.img.size() - QtCore.QSize(16, 16)
        pix = pix.scaled(target, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)

        self.img.setText("")
        self.img.setPixmap(pix)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        # Rerender on resize so the displayed pixmap uses the current size.
        super().resizeEvent(event)
        self._render()


class BlinkImagePanel(ImagePanel):
    """Animated blink panel: alternates between two images every interval_ms."""

    def __init__(self, title: str, min_w: int = 320, min_h: int = 240, interval_ms: int = 150) -> None:
        super().__init__(title, min_w=min_w, min_h=min_h)

        self._pix_a: Optional[QtGui.QPixmap] = None
        self._pix_b: Optional[QtGui.QPixmap] = None
        self._show_a: bool = True

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self._toggle)

    def set_interval_ms(self, interval_ms: int) -> None:
        """Update blink interval in milliseconds."""
        interval_ms = int(max(1, interval_ms))
        self._timer.setInterval(interval_ms)
        if self._timer.isActive():
            # Restart to apply immediately.
            self._timer.start()

    def set_images(self, path_a: Optional[str], path_b: Optional[str]) -> None:
        """Set two source images and start blinking if both are available."""

        self._timer.stop()
        self._pix_a = None
        self._pix_b = None
        self._show_a = True

        if path_a and os.path.isfile(path_a):
            pa = QtGui.QPixmap(path_a)
            if not pa.isNull():
                self._pix_a = pa

        if path_b and os.path.isfile(path_b):
            pb = QtGui.QPixmap(path_b)
            if not pb.isNull():
                self._pix_b = pb

        # Start only if both pixmaps are available.
        if self._pix_a is not None and self._pix_b is not None:
            self._timer.start()

        self._render_blink()

    def stop(self) -> None:
        """Stop blinking and clear the panel."""

        self._timer.stop()
        self._pix_a = None
        self._pix_b = None
        self._show_a = True
        self.img.setPixmap(QtGui.QPixmap())
        self.img.setText("—")

    def _toggle(self) -> None:
        self._show_a = not self._show_a
        self._render_blink()

    def _render_blink(self) -> None:
        pix = self._pix_a if self._show_a else self._pix_b

        # Fallback if only one is available.
        if pix is None:
            pix = self._pix_a or self._pix_b

        if pix is None:
            self.img.setPixmap(QtGui.QPixmap())
            self.img.setText("(missing)")
            return

        target = self.img.size() - QtCore.QSize(16, 16)
        spix = pix.scaled(target, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        self.img.setText("")
        self.img.setPixmap(spix)

    def get_preview_pixmap(self) -> Optional[QtGui.QPixmap]:
        """Return the currently displayed source pixmap for hover preview."""

        pix = self._pix_a if self._show_a else self._pix_b
        if pix is None:
            pix = self._pix_a or self._pix_b
        if pix is None:
            pm2 = self.img.pixmap()
            return pm2 if pm2 is not None else None
        return pix

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._render_blink()


class BarChartWidget(QtWidgets.QWidget):
    """Minimal in-app bar chart (no QtCharts dependency).

    Values are expected in [0..1]. The first bar (if present) can be used as an
    average score across selected metrics.
    """

    def __init__(self) -> None:
        super().__init__()
        self._labels: List[str] = []
        self._values: List[float] = []
        self._subtitle: str = ""
        self.setMinimumHeight(260)

    def set_data(self, labels: List[str], values: List[float], subtitle: str = "") -> None:
        self._labels = labels
        self._values = values
        self._subtitle = subtitle
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        r = self.rect().adjusted(12, 10, -12, -10)
        if not self._labels:
            p.setPen(self.palette().color(QtGui.QPalette.ColorRole.WindowText))
            p.drawText(r, QtCore.Qt.AlignmentFlag.AlignCenter, "No data (click Run to compute, then select metrics)")
            return

        # Title/subtitle.
        title_h = 22
        p.setPen(self.palette().color(QtGui.QPalette.ColorRole.WindowText))
        font = p.font()
        font.setPointSize(font.pointSize() + 1)
        font.setBold(True)
        p.setFont(font)
        p.drawText(
            QtCore.QRect(r.left(), r.top(), r.width(), title_h),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            "Selected Metrics",
        )

        font.setBold(False)
        font.setPointSize(max(8, font.pointSize() - 1))
        p.setFont(font)
        p.setPen(QtGui.QColor(170, 178, 192))
        p.drawText(
            QtCore.QRect(r.left(), r.top() + title_h, r.width(), 18),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            self._subtitle,
        )

        chart_top = r.top() + title_h + 22
        chart_rect = QtCore.QRect(r.left(), chart_top, r.width(), r.bottom() - chart_top)

        # Layout constants.
        left_pad = 32

        # X-axis label font + dynamic bottom padding (prevents clipped labels).
        label_font = QtGui.QFont(p.font())
        label_font.setPointSize(max(8, label_font.pointSize()))
        label_font.setBold(False)
        fm = QtGui.QFontMetrics(label_font)

        # Allow for up to two wrapped lines.
        bottom_pad = max(78, fm.height() * 2 + 34)
        top_pad = 10
        plot = QtCore.QRect(
            chart_rect.left() + left_pad,
            chart_rect.top() + top_pad,
            chart_rect.width() - left_pad,
            chart_rect.height() - top_pad - bottom_pad,
        )

        def _wrap_label(text: str, max_width: int, max_lines: int = 2) -> List[str]:
            """Wrap a label to at most max_lines and elide overflow."""

            text = (text or "").strip()
            if not text:
                return [""]

            words = text.split()
            lines: List[str] = []
            cur = ""
            for w in words:
                cand = w if not cur else f"{cur} {w}"
                if (fm.horizontalAdvance(cand) <= max_width) or (not cur):
                    cur = cand
                else:
                    lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)

            if len(lines) <= max_lines:
                return [fm.elidedText(l, QtCore.Qt.TextElideMode.ElideRight, max_width) for l in lines]

            head = lines[: max_lines - 1]
            tail = " ".join(lines[max_lines - 1 :])
            tail = fm.elidedText(tail, QtCore.Qt.TextElideMode.ElideRight, max_width)
            head = [fm.elidedText(l, QtCore.Qt.TextElideMode.ElideRight, max_width) for l in head]
            return head + [tail]

        # Axes.
        axis_color = QtGui.QColor(60, 66, 78)
        p.setPen(QtGui.QPen(axis_color, 1))
        p.drawLine(plot.bottomLeft(), plot.bottomRight())
        p.drawLine(plot.bottomLeft(), plot.topLeft())

        # Y-grid/ticks.
        tick_pen = QtGui.QPen(QtGui.QColor(44, 49, 58), 1)
        p.setPen(tick_pen)
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = int(plot.bottom() - t * plot.height())
            p.drawLine(plot.left(), y, plot.right(), y)
            p.setPen(QtGui.QColor(140, 145, 155))
            p.drawText(
                plot.left() - 30,
                y - 8,
                26,
                16,
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                f"{t:.2f}",
            )
            p.setPen(tick_pen)

        n = len(self._labels)
        if n == 0:
            return

        # Bar geometry.
        bar_gap = 10
        bar_w = max(10, int((plot.width() - bar_gap * (n - 1)) / n))
        total_w = bar_w * n + bar_gap * (n - 1)
        start_x = plot.left() + max(0, (plot.width() - total_w) // 2)

        # Color scale: 0.0 = full red, 1.0 = full green.
        def _scale_color(v: float) -> QtGui.QColor:
            v = float(np.clip(v, 0.0, 1.0))
            r = int(round(255.0 * (1.0 - v)))
            g = int(round(255.0 * v))
            return QtGui.QColor(r, g, 0)

        for i, (lab, val) in enumerate(zip(self._labels, self._values)):
            val = float(np.clip(val, 0.0, 1.0))
            bh = int(val * plot.height())
            x0 = start_x + i * (bar_w + bar_gap)
            y0 = plot.bottom() - bh
            rect = QtCore.QRect(x0, y0, bar_w, bh)

            # Bars.
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(_scale_color(val))
            p.drawRoundedRect(rect, 6, 6)

            # Value label: always anchored at the bottom of the bar.
            p.setPen(QtGui.QColor(235, 238, 245))
            vy = int(plot.bottom() - 18)
            p.drawText(QtCore.QRect(x0, vy, bar_w, 16), QtCore.Qt.AlignmentFlag.AlignCenter, f"{val:.2f}")

            # X label (wrapped + elided).
            p.save()
            p.setFont(label_font)
            p.setPen(QtGui.QColor(170, 178, 192))
            label_rect = QtCore.QRect(x0, plot.bottom() + 12, bar_w, bottom_pad - 12)
            max_w = max(10, bar_w)
            lines = _wrap_label(lab, max_w, max_lines=2)
            line_h = fm.height()
            y = label_rect.top()
            for li in lines:
                p.drawText(
                    QtCore.QRect(label_rect.left(), y, label_rect.width(), line_h),
                    QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop,
                    li,
                )
                y += line_h
            p.restore()


# =============================================================================
# Main Window
# =============================================================================

class MainWindow(QtWidgets.QMainWindow):
    """Main application window: scanning, selection, and visualization."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Stereo Evaluation")

        # Icon
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))

        self.resize(1400, 820)

        # Batch computation engine.
        self.engine = MetricEngine()
        self.engine.progress.connect(self._on_progress)
        self.engine.collection_done.connect(self._on_collection_done)
        self.engine.batch_finished.connect(self._on_batch_finished)
        self.engine.error.connect(self._on_engine_error)

        # Data model for the UI.
        self.collections: List[CollectionItem] = []
        self.collection_by_path: Dict[str, CollectionItem] = {}
        self.selected_collection_path: Optional[str] = None

        # ---------------------------------------------------------------------
        # Top bar: folder selector + actions
        # ---------------------------------------------------------------------

        self.input_edit = QtWidgets.QLineEdit()
        self.input_edit.setPlaceholderText("Select an input root folder (contains collection folders)")

        self.btn_browse = QtWidgets.QPushButton("Browse")
        self.btn_scan = QtWidgets.QPushButton("Scan")
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_run.setEnabled(False)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")

        # Smooth progress animation between worker completions.
        self._progress_anim_timer = QtCore.QTimer(self)
        self._progress_anim_timer.setInterval(50)
        self._progress_anim_timer.timeout.connect(self._tick_progress_animation)

        self._progress_anim_running: bool = False
        self._progress_anim_done: int = 0
        self._progress_anim_total: int = 0
        self._progress_anim_current_pct: float = 0.0
        self._progress_anim_avg_s: float = 1.0
        self._progress_anim_last_done_t: float = time.monotonic()

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(QtWidgets.QLabel("Input folder:"))
        top_row.addWidget(self.input_edit, 1)
        top_row.addWidget(self.btn_browse)
        top_row.addWidget(self.btn_scan)
        top_row.addWidget(self.btn_run)
        top_row.addWidget(self.progress)

        self.btn_browse.clicked.connect(self._browse_folder)
        self.btn_scan.clicked.connect(self._scan_folder)
        self.btn_run.clicked.connect(self._run_batch)
        # ---------------------------------------------------------------------
        # Left sidebar: collection list + filter
        # ---------------------------------------------------------------------

        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Filter collections…")

        self.list_collections = QtWidgets.QListWidget()
        self.list_collections.setObjectName("CollectionsList")
        self.list_collections.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_collections.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.list_collections.itemSelectionChanged.connect(self._on_collection_selected)
        self.filter_edit.textChanged.connect(self._apply_filter)

        sidebar_card = QtWidgets.QFrame()
        sidebar_card.setObjectName("Card")
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_card)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.setSpacing(10)
        sidebar_layout.addWidget(QtWidgets.QLabel("Collections"))
        sidebar_layout.addWidget(self.filter_edit)
        sidebar_layout.addWidget(self.list_collections, 1)

        # ---------------------------------------------------------------------
        # Right side: main content (segmented navigation + stacked pages)
        # ---------------------------------------------------------------------
        def _seg_button(text: str) -> QtWidgets.QToolButton:
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setCheckable(True)
            b.setProperty("segmented", True)
            b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            b.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
            return b

        # Main view switch (Overview / Metrics)
        self.btn_view_overview = _seg_button("Overview")
        self.btn_view_metrics = _seg_button("Metrics")

        view_group = QtWidgets.QButtonGroup(self)
        view_group.setExclusive(True)
        view_group.addButton(self.btn_view_overview, 0)
        view_group.addButton(self.btn_view_metrics, 1)
        self.btn_view_overview.setChecked(True)

        view_bar = QtWidgets.QWidget()
        view_bar_layout = QtWidgets.QHBoxLayout(view_bar)
        view_bar_layout.setContentsMargins(0, 0, 0, 0)
        view_bar_layout.setSpacing(8)
        view_bar_layout.addWidget(self.btn_view_overview)
        view_bar_layout.addWidget(self.btn_view_metrics)
        view_bar_layout.addStretch(1)

        self.main_stack = QtWidgets.QStackedWidget()

        # ---------------- Overview page ----------------

        # Large image panels.
        self.panel_right = ImagePanel("Right (GT)", min_w=520, min_h=320)
        self.panel_gen = ImagePanel("Generated", min_w=520, min_h=320)
        self.panel_blink = BlinkImagePanel("Blink (GT ↔ Generated)", min_w=520, min_h=320, interval_ms=150)
        self.panel_left = ImagePanel("Left", min_w=520, min_h=320)

        # Metadata view (borderless; the surrounding card provides the frame).
        self.meta_view = QtWidgets.QTextEdit()
        self.meta_view.setObjectName("MetaView")
        self.meta_view.setReadOnly(True)
        self.meta_view.setPlaceholderText("meta.json will be shown here")
        self.meta_view.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        try:
            self.meta_view.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        except Exception:
            pass

        # Hide internal titles; selector buttons already label the views.
        for _p in (self.panel_right, self.panel_gen, self.panel_blink, self.panel_left):
            try:
                _p.title.setVisible(False)
            except Exception:
                pass

        # Image selector (segmented buttons) + stacked image views.
        self.image_stack = QtWidgets.QStackedWidget()
        self.image_stack.addWidget(self.panel_right)  # 0
        self.image_stack.addWidget(self.panel_gen)    # 1
        self.image_stack.addWidget(self.panel_blink)  # 2
        self.image_stack.addWidget(self.panel_left)   # 3

        self.btn_img_right = _seg_button("Right (GT)")
        self.btn_img_gen = _seg_button("Generated")
        self.btn_img_blink = _seg_button("Blink")
        self.btn_img_left = _seg_button("Left")

        img_group = QtWidgets.QButtonGroup(self)
        img_group.setExclusive(True)
        img_group.addButton(self.btn_img_right, 0)
        img_group.addButton(self.btn_img_gen, 1)
        img_group.addButton(self.btn_img_blink, 2)
        img_group.addButton(self.btn_img_left, 3)
        self.btn_img_right.setChecked(True)

        img_bar = QtWidgets.QWidget()
        img_bar_layout = QtWidgets.QHBoxLayout(img_bar)
        img_bar_layout.setContentsMargins(0, 0, 0, 0)
        img_bar_layout.setSpacing(8)
        img_bar_layout.addWidget(self.btn_img_right)
        img_bar_layout.addWidget(self.btn_img_gen)
        img_bar_layout.addWidget(self.btn_img_blink)
        img_bar_layout.addWidget(self.btn_img_left)

        # Blink speed slider (0.10s .. 1.00s)
        self.blink_speed_label = QtWidgets.QLabel("Blink: 0.15s")
        self.blink_speed_label.setStyleSheet("color: #AAB2C0;")

        self.blink_speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.blink_speed_slider.setRange(100, 1000)  # ms
        self.blink_speed_slider.setSingleStep(10)
        self.blink_speed_slider.setPageStep(50)
        self.blink_speed_slider.setValue(150)
        self.blink_speed_slider.setFixedWidth(160)
        self.blink_speed_slider.setToolTip("Blink interval (seconds)")

        def _on_blink_speed(val_ms: int) -> None:
            self.panel_blink.set_interval_ms(val_ms)
            self.blink_speed_label.setText(f"Blink: {val_ms / 1000.0:.2f}s")

        self.blink_speed_slider.valueChanged.connect(_on_blink_speed)

        img_bar_layout.addSpacing(6)
        img_bar_layout.addWidget(self.blink_speed_label)
        img_bar_layout.addWidget(self.blink_speed_slider)
        img_bar_layout.addStretch(1)

        # Left side (images)
        overview_left = QtWidgets.QWidget()
        overview_left_layout = QtWidgets.QVBoxLayout(overview_left)
        overview_left_layout.setContentsMargins(12, 12, 0, 12)
        overview_left_layout.setSpacing(10)
        overview_left_layout.addWidget(img_bar)
        overview_left_layout.addWidget(self.image_stack, 1)

        # Right side (meta)
        overview_right = QtWidgets.QWidget()
        overview_right_layout = QtWidgets.QVBoxLayout(overview_right)
        overview_right_layout.setContentsMargins(12, 12, 12, 12)
        overview_right_layout.setSpacing(8)

        meta_title = QtWidgets.QLabel("Meta (meta.json)")
        meta_title.setStyleSheet("font-weight: 600;")
        overview_right_layout.addWidget(meta_title)
        overview_right_layout.addWidget(self.meta_view, 1)

        overview_split = QtWidgets.QSplitter()
        overview_split.setOrientation(QtCore.Qt.Orientation.Horizontal)
        overview_split.setObjectName("OverviewSplit")
        overview_split.setHandleWidth(1)
        overview_split.setChildrenCollapsible(False)
        overview_split.addWidget(overview_left)
        overview_split.addWidget(overview_right)
        overview_split.setStretchFactor(0, 1)
        overview_split.setStretchFactor(1, 0)
        overview_split.setSizes([1000, 380])

        overview_card = QtWidgets.QFrame()
        overview_card.setObjectName("OverviewCard")
        overview_card_layout = QtWidgets.QVBoxLayout(overview_card)
        overview_card_layout.setContentsMargins(0, 0, 0, 0)
        overview_card_layout.setSpacing(0)
        overview_card_layout.addWidget(overview_split, 1)

        overview_page = QtWidgets.QWidget()
        overview_page_layout = QtWidgets.QVBoxLayout(overview_page)
        overview_page_layout.setContentsMargins(0, 0, 0, 0)
        overview_page_layout.setSpacing(0)
        overview_page_layout.addWidget(overview_card, 1)

        # ---------------- Metrics page ----------------

        self.chart = BarChartWidget()

        self.metric_filter = QtWidgets.QLineEdit()
        self.metric_filter.setPlaceholderText("Filter metrics…")

        self.btn_sel_all = QtWidgets.QPushButton("Select all")
        self.btn_sel_none = QtWidgets.QPushButton("Select none")
        self.btn_sel_reco = QtWidgets.QPushButton("Recommended")

        self.btn_sel_all.clicked.connect(lambda: self._set_all_checks(True))
        self.btn_sel_none.clicked.connect(lambda: self._set_all_checks(False))
        self.btn_sel_reco.clicked.connect(self._select_recommended)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)
        btn_row.addWidget(self.btn_sel_reco)
        btn_row.addWidget(self.btn_sel_all)
        btn_row.addWidget(self.btn_sel_none)
        btn_row.addStretch(1)

        # Checkbox list in a scroll area (borderless; surrounding card frames it).
        self.metric_checks: Dict[str, QtWidgets.QCheckBox] = {}
        self.metric_check_container = QtWidgets.QWidget()
        self.metric_check_layout = QtWidgets.QVBoxLayout(self.metric_check_container)
        self.metric_check_layout.setContentsMargins(0, 0, 0, 0)
        self.metric_check_layout.setSpacing(6)
        self.metric_check_layout.addStretch(1)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.metric_check_container)
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self.metric_filter.textChanged.connect(self._filter_metrics)

        # Table showing full metric details for the selected collection.
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Metric", "Score (0..1)", "Raw", "Available", "Notes"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.setCornerButtonEnabled(False)
        self.table.setObjectName("MetricsTable")

        # Wrap the table in its own rounded frame so the header corners are properly rounded
        self.table_card = QtWidgets.QFrame()
        self.table_card.setObjectName("TableCard")
        table_card_l = QtWidgets.QVBoxLayout(self.table_card)
        table_card_l.setContentsMargins(0, 0, 0, 0)
        table_card_l.setSpacing(0)
        table_card_l.addWidget(self.table)
        self.table.verticalHeader().setVisible(False)

        self.status = QtWidgets.QLabel("Scan a folder, choose metrics, then click Run.")
        self.status.setStyleSheet("color: #AAB2C0;")

        # Left metrics card
        metrics_left_card = QtWidgets.QFrame()
        metrics_left_card.setObjectName("Card")
        metrics_left = QtWidgets.QVBoxLayout(metrics_left_card)
        metrics_left.setContentsMargins(12, 12, 12, 12)
        metrics_left.setSpacing(10)
        lab = QtWidgets.QLabel("Metrics to visualize")
        lab.setStyleSheet("font-weight: 600;")
        metrics_left.addWidget(lab)
        metrics_left.addWidget(self.metric_filter)
        metrics_left.addLayout(btn_row)
        metrics_left.addWidget(self.scroll, 1)

        # Right metrics card
        metrics_right_card = QtWidgets.QFrame()
        metrics_right_card.setObjectName("Card")
        metrics_right = QtWidgets.QVBoxLayout(metrics_right_card)
        metrics_right.setContentsMargins(12, 12, 12, 12)
        metrics_right.setSpacing(10)
        metrics_right.addWidget(self.chart)
        metrics_right.addWidget(self.table_card, 1)
        metrics_right.addWidget(self.status)

        metrics_page = QtWidgets.QWidget()
        metrics_page_layout = QtWidgets.QHBoxLayout(metrics_page)
        metrics_page_layout.setContentsMargins(0, 0, 0, 0)
        metrics_page_layout.setSpacing(14)
        metrics_page_layout.addWidget(metrics_left_card, 0)
        metrics_page_layout.addWidget(metrics_right_card, 1)

        self.main_stack.addWidget(overview_page)  # index 0
        self.main_stack.addWidget(metrics_page)   # index 1

        # Hook view switching.
        def _set_main_index(idx: int) -> None:
            self.main_stack.setCurrentIndex(idx)

        view_group.idClicked.connect(_set_main_index)
        img_group.idClicked.connect(self.image_stack.setCurrentIndex)

        # Container used by the global splitter (collections | content)
        self.main_container = QtWidgets.QWidget()
        main_container_layout = QtWidgets.QVBoxLayout(self.main_container)
        main_container_layout.setContentsMargins(0, 0, 0, 0)
        main_container_layout.setSpacing(10)
        main_container_layout.addWidget(view_bar)
        main_container_layout.addWidget(self.main_stack, 1)

# ---------------------------------------------------------------------
        # Global split layout: left list + right tabs
        # ---------------------------------------------------------------------

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        splitter.setObjectName("MainSplit")
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(sidebar_card)
        splitter.addWidget(self.main_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1080])

        central = QtWidgets.QWidget()
        main = QtWidgets.QVBoxLayout(central)
        main.setContentsMargins(14, 14, 14, 14)
        main.setSpacing(12)
        main.addLayout(top_row)
        main.addWidget(splitter, 1)
        self.setCentralWidget(central)

        # Build metric selection UI and set a sensible default.
        self._build_metric_checkboxes()
        self._select_recommended()

    # -------------------------------------------------------------------------
    # Folder scanning
    # -------------------------------------------------------------------------

    def _browse_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select input folder")
        if folder:
            self.input_edit.setText(folder)

    def _scan_folder(self) -> None:
        root = self.input_edit.text().strip()
        if not root:
            QtWidgets.QMessageBox.information(self, "Input folder", "Please choose an input folder.")
            return
        if not os.path.isdir(root):
            QtWidgets.QMessageBox.warning(self, "Input folder", "The selected path is not a folder.")
            return

        self.collections = find_collections(root)
        self.collection_by_path = {c.path: c for c in self.collections}
        self._refresh_collection_list()

        # Clear previous results to avoid mixing stale and fresh runs.
        self.engine.clear_cache()
        self._clear_results_views()

        self.btn_run.setEnabled(len(self.collections) > 0)
        self.status.setText(f"Found {len(self.collections)} collections. Click Run to compute batch metrics.")

        if self.collections:
            self.list_collections.setCurrentRow(0)
        else:
            QtWidgets.QMessageBox.information(
                self,
                "No collections found",
                "No collection folders found. Expected folders containing: left.jpg, right.jpg, generated.jpg, meta.json.",
            )

    def _refresh_collection_list(self) -> None:
        # Block signals to avoid selection callbacks during list rebuild.
        self.list_collections.blockSignals(True)
        self.list_collections.clear()
        for c in self.collections:
            it = QtWidgets.QListWidgetItem(c.name)
            it.setData(QtCore.Qt.ItemDataRole.UserRole, c.path)
            self.list_collections.addItem(it)
        self.list_collections.blockSignals(False)
        self._apply_filter()

    def _apply_filter(self) -> None:
        q = self.filter_edit.text().strip().lower()
        for i in range(self.list_collections.count()):
            it = self.list_collections.item(i)
            it.setHidden(bool(q) and q not in it.text().lower())

    # -------------------------------------------------------------------------
    # Batch run
    # -------------------------------------------------------------------------

    def _start_progress_animation(self, total: int) -> None:
        self._progress_anim_running = True
        self._progress_anim_done = 0
        self._progress_anim_total = int(max(0, total))
        self._progress_anim_current_pct = 0.0
        self._progress_anim_avg_s = 1.0
        self._progress_anim_last_done_t = time.monotonic()

        if self._progress_anim_total > 0:
            self._progress_anim_timer.start()
        else:
            self._progress_anim_timer.stop()

    def _stop_progress_animation(self) -> None:
        self._progress_anim_running = False
        self._progress_anim_timer.stop()

    def _tick_progress_animation(self) -> None:
        if (not self._progress_anim_running) or self._progress_anim_total <= 0:
            return

        done = int(max(0, self._progress_anim_done))
        total = int(max(1, self._progress_anim_total))
        now = time.monotonic()

        # Smoothly advance within the current "collection slot" without running ahead.
        avg = max(0.15, float(self._progress_anim_avg_s))
        frac_next = min(0.95, (now - float(self._progress_anim_last_done_t)) / avg)

        virtual_done = float(done) + float(frac_next)
        if done >= total:
            virtual_done = float(total)

        # Never display 100% until the batch is actually complete.
        target_pct = 100.0 * (virtual_done / float(total))
        hard_min = 100.0 * (float(done) / float(total))

        target_pct = max(target_pct, hard_min)
        if done < total:
            target_pct = min(target_pct, 99.0)

        # Approach target with many small steps.
        if target_pct > self._progress_anim_current_pct:
            diff = target_pct - self._progress_anim_current_pct
            step = min(2.0, max(0.15, diff * 0.25))
            self._progress_anim_current_pct = min(target_pct, self._progress_anim_current_pct + step)

        disp = int(round(self._progress_anim_current_pct))
        disp = max(disp, int(round(hard_min)))
        disp = min(disp, 99 if done < total else 100)

        self.progress.setValue(disp)
        self.progress.setFormat(f"{disp}%")

    def _run_batch(self) -> None:
        if not self.collections:
            return

        self._set_busy(True)
        self.progress.setValue(0)
        self.progress.setFormat("0%")
        self._start_progress_animation(len(self.collections))
        self.status.setText("Running batch metric computation…")
        self.engine.run_all(self.collections)

    def _set_busy(self, busy: bool) -> None:
        """Enable/disable UI controls during computation."""

        self.btn_run.setEnabled(not busy and len(self.collections) > 0)
        self.btn_scan.setEnabled(not busy)
        self.btn_browse.setEnabled(not busy)
        self.list_collections.setEnabled(not busy)
        self.metric_filter.setEnabled(not busy)

        for cb in self.metric_checks.values():
            cb.setEnabled(not busy)

        self.btn_sel_all.setEnabled(not busy)
        self.btn_sel_none.setEnabled(not busy)
        self.btn_sel_reco.setEnabled(not busy)

    @QtCore.Slot(int, int)
    def _on_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self._stop_progress_animation()
            self.progress.setValue(0)
            self.progress.setFormat("Idle")
            return

        now = time.monotonic()

        # Update timing estimate when a collection finishes (or fails).
        if done > self._progress_anim_done:
            dt = max(0.05, now - float(self._progress_anim_last_done_t))
            self._progress_anim_avg_s = 0.8 * float(self._progress_anim_avg_s) + 0.2 * float(dt)
            self._progress_anim_last_done_t = now

        self._progress_anim_done = int(max(0, done))
        self._progress_anim_total = int(max(1, total))

        if done >= total:
            self._stop_progress_animation()
            self.progress.setValue(100)
            self.progress.setFormat("100%")
            return

        # Keep the display responsive immediately after updates.
        self._tick_progress_animation()

    @QtCore.Slot(str)
    def _on_collection_done(self, collection_path: str) -> None:
        # Refresh current selection if it matches.
        if self.selected_collection_path == collection_path:
            self._render_results_for_selected_collection()

    @QtCore.Slot()
    def _on_batch_finished(self) -> None:
        self._stop_progress_animation()
        self.progress.setValue(100)
        self.progress.setFormat("100%")
        self._set_busy(False)
        self.status.setText("Batch finished. Select a collection to inspect results.")
        self._render_results_for_selected_collection()

    @QtCore.Slot(str, str)
    def _on_engine_error(self, collection_path: str, message: str) -> None:
        # Non-fatal; still counts toward progress.
        print(f"Error for {collection_path}:\n{message}")

    # -------------------------------------------------------------------------
    # Selection handling
    # -------------------------------------------------------------------------

    def _on_collection_selected(self) -> None:
        items = self.list_collections.selectedItems()
        if not items:
            self.selected_collection_path = None
            self._clear_overview()
            self._clear_results_views()
            return

        path = items[0].data(QtCore.Qt.ItemDataRole.UserRole)
        if not path:
            return

        self.selected_collection_path = str(path)
        col = self.collection_by_path[self.selected_collection_path]

        # Overview images.
        self.panel_left.set_image(col.left_path)
        self.panel_right.set_image(col.right_path)
        self.panel_gen.set_image(col.gen_path)
        self.panel_blink.set_images(col.right_path, col.gen_path)

        # Metadata.
        self.meta_view.setPlainText(self._format_meta(_load_meta(col.meta_path)))

        # Metrics.
        self._render_results_for_selected_collection()

    def _format_meta(self, meta: Dict[str, Any]) -> str:
        """Render a small subset of meta.json keys in a readable form."""

        if not meta:
            return "(no meta.json or failed to parse)"

        keys = [
            "base_camera_name",
            "width",
            "height",
            "baseline_m",
            "fx",
            "fy",
            "cx",
            "cy",
            "fov_deg",
            "focal_mm",
        ]

        lines = []
        for k in keys:
            if k in meta:
                lines.append(f"{k}: {meta[k]}")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Metric selection (checkboxes)
    # -------------------------------------------------------------------------

    def _build_metric_checkboxes(self) -> None:
        """Build the checkbox list of available metrics."""

        # Remove existing widgets.
        while self.metric_check_layout.count():
            item = self.metric_check_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self.metric_checks.clear()

        # Show only metrics that are currently available.
        metrics_sorted = sorted(_METRICS, key=lambda m: (m.is_stereo_extra, m.display_name.lower()))
        available_metrics: List[Metric] = []
        for m in metrics_sorted:
            ok, _reason = m.is_available()
            if ok:
                available_metrics.append(m)

        def add_header(text: str) -> None:
            lab = QtWidgets.QLabel(text)
            lab.setStyleSheet("font-weight: 700; color: #CDD3E0;")
            self.metric_check_layout.addWidget(lab)

        img_metrics = [m for m in available_metrics if not m.is_stereo_extra]
        stereo_metrics = [m for m in available_metrics if m.is_stereo_extra]

        if img_metrics:
            add_header("Image similarity metrics")
            for m in img_metrics:
                cb = QtWidgets.QCheckBox(m.display_name)
                cb.setToolTip(m.description)
                cb.stateChanged.connect(self._on_metric_selection_changed)
                self.metric_checks[m.key] = cb
                self.metric_check_layout.addWidget(cb)

        if stereo_metrics:
            add_header("Stereo extras (use left image)")
            for m in stereo_metrics:
                cb = QtWidgets.QCheckBox(m.display_name)
                cb.setToolTip(m.description)
                cb.stateChanged.connect(self._on_metric_selection_changed)
                self.metric_checks[m.key] = cb
                self.metric_check_layout.addWidget(cb)

        self.metric_check_layout.addStretch(1)

    def _filter_metrics(self) -> None:
        """Filter the checkbox list by a case-insensitive substring match."""

        q = self.metric_filter.text().strip().lower()
        for key, cb in self.metric_checks.items():
            cb.setVisible((not q) or (q in cb.text().lower()))

    def _set_all_checks(self, value: bool) -> None:
        """Select or deselect all visible metric checkboxes."""

        for cb in self.metric_checks.values():
            cb.blockSignals(True)
            cb.setChecked(value)
            cb.blockSignals(False)
        self._on_metric_selection_changed()

    def _select_recommended(self) -> None:
        """Select a practical default subset of metrics."""

        recommended = {"ms_ssim", "lpips", "dists", "psnr", "deltae2000", "vifp", "fsim", "stereo_disp_epe"}
        for key, cb in self.metric_checks.items():
            cb.blockSignals(True)
            cb.setChecked(key in recommended)
            cb.blockSignals(False)
        self._on_metric_selection_changed()

    def _selected_metric_keys(self) -> List[str]:
        """Return selected metric keys (checked and visible)."""

        return [k for k, cb in self.metric_checks.items() if cb.isChecked() and cb.isVisible()]

    def _on_metric_selection_changed(self) -> None:
        # Update chart immediately when selection changes.
        self._render_bar_chart()

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def _clear_overview(self) -> None:
        self.panel_left.img.setText("—")
        self.panel_right.img.setText("—")
        self.panel_gen.img.setText("—")
        self.panel_blink.stop()
        self.meta_view.clear()

    def _clear_results_views(self) -> None:
        self.table.setRowCount(0)
        self.chart.set_data([], [])

    def _render_results_for_selected_collection(self) -> None:
        if not self.selected_collection_path:
            self._clear_results_views()
            return

        cached = self.engine.get_cached(self.selected_collection_path)
        if cached is None:
            self.table.setRowCount(0)
            self.chart.set_data([], [])
            self.status.setText("No results yet for this collection. Click Run to compute batch metrics.")
            return

        self._render_table(list(cached.values()))
        self._render_bar_chart()

    def _render_table(self, results: List[MetricResult]) -> None:
        """Populate the metric results table."""

        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(results))

        for r, res in enumerate(results):
            name = QtWidgets.QTableWidgetItem(res.display_name)

            score = "—" if res.score_0_1 is None else f"{res.score_0_1:.4f}"
            raw = "—" if res.raw_value is None else ("∞" if not np.isfinite(res.raw_value) else f"{res.raw_value:.6g}")
            avail = "Yes" if res.available else "No"
            notes = res.details or ""

            it_score = QtWidgets.QTableWidgetItem(score)
            it_raw = QtWidgets.QTableWidgetItem(raw)
            it_avail = QtWidgets.QTableWidgetItem(avail)
            it_notes = QtWidgets.QTableWidgetItem(notes)

            it_score.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            it_raw.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

            self.table.setItem(r, 0, name)
            self.table.setItem(r, 1, it_score)
            self.table.setItem(r, 2, it_raw)
            self.table.setItem(r, 3, it_avail)
            self.table.setItem(r, 4, it_notes)

        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)

    def _render_bar_chart(self) -> None:
        """Render the bar chart for the currently selected collection."""

        if not self.selected_collection_path:
            self.chart.set_data([], [])
            return

        cached = self.engine.get_cached(self.selected_collection_path)
        if cached is None:
            self.chart.set_data([], [])
            return

        selected_keys = [k for k, cb in self.metric_checks.items() if cb.isChecked()]
        if not selected_keys:
            self.chart.set_data([], [], subtitle="No metrics selected")
            return

        metric_by_key = {m.key: m for m in _METRICS}

        include_avg = len(selected_keys) > 1
        labels: List[str] = []
        values: List[float] = []

        vals_for_avg: List[float] = []

        # If multiple metrics are selected, the first bar is the average.
        if include_avg:
            labels.append("Average")
            values.append(0.0)

        for k in selected_keys:
            m = metric_by_key.get(k)
            display = m.display_name if m else k
            res = cached.get(k)

            if res is None or (not res.available) or (res.score_0_1 is None):
                v = 0.0
            else:
                v = float(np.clip(res.score_0_1, 0.0, 1.0))
                vals_for_avg.append(v)

            labels.append(display)
            values.append(v)

        if include_avg:
            avg = float(np.mean(vals_for_avg)) if vals_for_avg else 0.0
            values[0] = float(np.clip(avg, 0.0, 1.0))
            subtitle = f"Average over {len(vals_for_avg)}/{len(selected_keys)} available selected metrics"
        else:
            subtitle = "Single metric selected (average bar omitted)"

        self.chart.set_data(labels, values, subtitle=subtitle)


# =============================================================================
# Main entry point
# =============================================================================

def main() -> int:
    """Qt application entry point."""

    app = QtWidgets.QApplication(sys.argv)
    apply_modern_dark_palette(app)

    # High-DPI pixmaps improve image scaling on modern displays.
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

# TODO write tests ?!