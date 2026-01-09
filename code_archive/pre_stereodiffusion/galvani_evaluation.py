# baseline_eval_tokens_only.py
import os, math, cv2, torch, numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# -------- Helpers --------
def _to_gray_np(img: Image.Image):
    return np.asarray(img.convert("L"))

def _center_crop(arr, frac=0.6):
    H, W = arr.shape[:2]
    ch, cw = int(H*frac), int(W*frac)
    y0 = (H - ch)//2; x0 = (W - cw)//2
    return arr[y0:y0+ch, x0:x0+cw]

def _compute_flow(left_img: Image.Image, right_img: Image.Image, crop_frac=0.6):
    """Dense optical flow L->R (Farnebäck). Rückgabe: robuste Disparitäts-Metriken."""
    gL = _to_gray_np(left_img)
    gR = _to_gray_np(right_img)

    # leichtes Downscale für Robustheit
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
    u = flow[..., 0]; v = flow[..., 1]

    u_c = _center_crop(u, frac=crop_frac)
    v_c = _center_crop(v, frac=crop_frac)

    # Texturmaske
    gx = cv2.Sobel(gL, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gL, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx**2 + gy**2)
    gm_c = _center_crop(gm, frac=crop_frac)
    thr = np.percentile(gm_c, 40)
    m = gm_c > max(1.0, thr)

    u_m = u_c[m]; v_m = v_c[m]
    if u_m.size == 0:
        u_m = u_c.ravel(); v_m = v_c.ravel()

    med_u = float(np.median(u_m))
    iqr_u = float(np.percentile(u_m, 75) - np.percentile(u_m, 25))
    mean_abs_u = float(np.mean(np.abs(u_m)))
    frac_pos = float(np.mean(u_m > 0))
    vert_ratio = float(np.median(np.abs(v_m)) / (np.median(np.abs(u_m)) + 1e-6))

    return {
        "median_u": med_u,
        "iqr_u": iqr_u,
        "mean_abs_u": mean_abs_u,
        "frac_pos": frac_pos,
        "vert_over_horiz": vert_ratio,
        "num_samples": int(u_m.size),
    }

def _prompt(b_cm: int, side: str):
    # exakt deine gelernten Tokens, ohne Zusatztext/FOV
    return f"<B_{int(b_cm):02d}> <{side.upper()}>"

def set_lora_adapters(pipelines, adapter_name="stereo", lora_scale=1.0):
    for p in pipelines:
        try:
            p.set_adapters([adapter_name])
            p.set_lora_scale(lora_scale)
        except Exception:
            # Falls full-UNet geladen wurde, sind Adapter-Calls harmlos zu ignorieren
            pass

# -------- Haupt-Eval --------
@torch.inference_mode()
def evaluate_baseline_control(
    txt2img,
    img2img,
    baselines_cm=(6, 8, 10, 12, 14, 16, 18, 20, 30, 40),  # nur verwenden, wenn im Training vorhanden
    seeds=(11, 22, 33, 44),
    steps=28,
    guidance=5.0,
    strength=0.25,
    W=512,
    H=512,
    out_dir="/content/baseline_eval",
    save_images=True,
    plot=True,
    lora_scale=1.0,
    adapter_name="stereo",
):
    os.makedirs(out_dir, exist_ok=True)

    # Adapter & Scale einmal setzen (no-op wenn Full-UNet)
    set_lora_adapters([txt2img, img2img], adapter_name=adapter_name, lora_scale=lora_scale)

    records = []
    device = txt2img.device if hasattr(txt2img, "device") else "cuda"
    for seed in seeds:
        g = torch.Generator(device=device).manual_seed(int(seed))
        for b in baselines_cm:
            pL = _prompt(b, "left")
            pR = _prompt(b, "right")

            # Left via txt2img
            left = txt2img(
                prompt=pL,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=H,
                width=W,
                generator=g,
            ).images[0]

            # Right via img2img (gleiches Motiv, Parallax über Side-Token)
            right = img2img(
                prompt=pR,
                image=left,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=g,
            ).images[0]

            if save_images:
                base = f"seed{seed}_B{b:02d}"
                left.save(os.path.join(out_dir, f"{base}_L.png"))
                right.save(os.path.join(out_dir, f"{base}_R.png"))

            m = _compute_flow(left, right, crop_frac=0.6)
            rec = {"seed": int(seed), "baseline_cm": int(b)}
            rec.update(m)
            records.append(rec)

    df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(out_dir, "baseline_eval_metrics.csv")
    df.to_csv(csv_path, index=False)

    # Aggregation über Seeds
    agg = df.groupby("baseline_cm").agg(
        median_u_med=("median_u", "median"),
        median_u_iqr=("median_u", lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
        vert_over_horiz_med=("vert_over_horiz", "median"),
    ).reset_index()

    # Lineares Fit: median_u_med ~ a * baseline + b
    x = agg["baseline_cm"].values.astype(np.float32)
    y = agg["median_u_med"].values.astype(np.float32)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    y_hat = a * x + b

    # R^2 und Pearson r
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2) + 1e-9)
    r2 = 1.0 - ss_res / ss_tot
    pear = float(np.corrcoef(x, y)[0, 1])
    mono_viol = int(np.sum(np.diff(y) < -1e-6))

    summary = {
        "slope_pix_per_cm": float(a),
        "intercept_pix": float(b),
        "R2": r2,
        "pearson_r": pear,
        "monotonicity_violations": mono_viol,
        "median_vert_over_horiz": float(agg["vert_over_horiz_med"].median()),
        "num_seeds": int(len(seeds)),
        "num_baselines": int(len(baselines_cm)),
        "csv": csv_path,
    }

    if plot:
        plt.figure(figsize=(6, 4))
        # per-seed Kurven
        for s in seeds:
            sdf = df[df.seed == s]
            plt.plot(sdf["baseline_cm"], sdf["median_u"], "o-", alpha=0.35)
        # Aggregat + Fit
        plt.plot(x, y, "ko", label="median over seeds")
        plt.plot(x, y_hat, "-", label=f"fit: y={a:.3f}x+{b:.3f}, R²={r2:.3f}")
        plt.xlabel("Requested baseline (cm)")
        plt.ylabel("Measured median horizontal flow (pixels)")
        plt.title(f"Baseline control (guidance={guidance}, strength={strength}, lora={lora_scale})")
        plt.grid(True, alpha=0.3); plt.legend()
        plot_path = os.path.join(out_dir, "baseline_eval_plot.png")
        plt.tight_layout(); plt.savefig(plot_path, dpi=160)
        summary["plot"] = plot_path

    print(
        f"[EVAL] slope={summary['slope_pix_per_cm']:.3f} px/cm | "
        f"R²={summary['R2']:.3f} | r={summary['pearson_r']:.3f} | "
        f"monotonicity_violations={summary['monotonicity_violations']} | "
        f"vert/horiz={summary['median_vert_over_horiz']:.3f}"
    )
    print(f"[EVAL] CSV:  {summary['csv']}")
    if "plot" in summary: print(f"[EVAL] Plot: {summary['plot']}")

    return df, agg, summary
