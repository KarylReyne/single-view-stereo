import cv2
import numpy as np
from PIL import Image


def _to_gray_np(img: Image.Image):
    return np.asarray(img.convert("L"))


def _center_crop(arr, frac=0.6):
    H, W = arr.shape[:2]
    ch, cw = int(H*frac), int(W*frac)
    y0 = (H - ch)//2; x0 = (W - cw)//2
    return arr[y0:y0+ch, x0:x0+cw]


def estimate_disparity_px(left_path, right_path, crop_frac=0.6):
    # Laden
    L = Image.open(left_path)
    R = Image.open(right_path)

    gL = _to_gray_np(L)
    gR = _to_gray_np(R)

    # optional: etwas runter skalieren für Robustheit
    scale = 0.75
    if scale != 1.0:
        gL = cv2.resize(gL, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gR = cv2.resize(gR, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    flow = cv2.calcOpticalFlowFarneback(
        gL, gR, None,
        pyr_scale=0.5, levels=3, winsize=21,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    u = flow[..., 0]  # horizontale Verschiebung
    v = flow[..., 1]

    # Zentrum croppen
    u_c = _center_crop(u, frac=crop_frac)
    v_c = _center_crop(v, frac=crop_frac)

    # Low-Texture per Gradienten rausfiltern
    gx = cv2.Sobel(gL, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gL, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx**2 + gy**2)
    gm_c = _center_crop(gm, frac=crop_frac)
    thr = np.percentile(gm_c, 40)
    mask = gm_c > max(1.0, thr)

    u_m = u_c[mask]
    if u_m.size == 0:
        u_m = u_c.ravel()

    median_u = float(np.median(u_m))        # vorzeichenbehaftet
    mean_abs_u = float(np.mean(np.abs(u_m)))  # immer positiv

    return {
        "median_u_px": median_u,
        "mean_abs_u_px": mean_abs_u,
        "num_samples": int(u_m.size),
    }