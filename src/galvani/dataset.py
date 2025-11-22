from torch.utils.data import Dataset
import os
import json
import math
from PIL import Image


# Describes stereo dataset
class StereoScenesDataset(Dataset):
    def __init__(self, root_dir, size=(384, 384)):
        self.root_dir = root_dir
        self.size = size
        # collecting all scenes
        self.samples = self._collect()

    # returns baseline/fov for a meta file
    def _read_meta(self, meta_path):
        baseline, fov = 0.08, 60.0
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                m = json.load(f)

        if "baseline_m" in m:
            baseline = float(m["baseline_m"])

        if "fov_deg" in m:
            fov = float(m["fov_deg"])

        # Fallbacks
        # No FOV in meta file
        if (m.get("fov_deg") is None) and "fx" in m and "width" in m:
            # calculate fov from width and focal length
            fov = math.degrees(
                2.0 * math.atan((float(m["width"]) * 0.5) / float(m["fx"]))
            )

        # No baseline in meta file
        if (m.get("baseline_m") is None) and "cam_left" in m and "cam_right" in m:
            # calculate baseline from x coordinates of both cameras
            try:
                tx_l = float(m["cam_left"]["matrix_world"][0][3])
                tx_r = float(m["cam_right"]["matrix_world"][0][3])
                baseline = abs(tx_r - tx_l)
            except Exception:
                pass
        return max(0.01, min(1.0, baseline)), max(20.0, min(120.0, fov))

    def _find(self, d, scene, tag):
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(d, f"{scene}_{tag}{ext}")
            if os.path.exists(p):
                return p
        return None

    def _collect(self):
        S = []
        # collecting all image
        for scene in sorted(os.listdir(self.root_dir)):
            d = os.path.join(self.root_dir, scene)
            if not os.path.isdir(d):
                continue
            # path left, right
            L = self._find(d, scene, "left")
            R = self._find(d, scene, "right")
            # skip if not left and right
            if not (L and R):
                continue
            # meta
            meta = os.path.join(d, f"{scene}_meta.json")
            # baseline (ignore fov)
            B, _ = self._read_meta(meta) if os.path.exists(meta) else (0.08, 60.0)

            # ignore too large baselines
            if B > 0.20:
                continue

            btag = f"<B_{int(round(B*100)):02d}>"

            S += [
                {"image_path": L, "prompt": f"{btag} <LEFT>", "btag": btag},
                {"image_path": R, "prompt": f"{btag} <RIGHT>", "btag": btag},
            ]

        if not S:
            raise RuntimeError(f"No scene pairs found under {self.root_dir}")
        return S

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB").resize(self.size)
        return {"image": img, "prompt": s["prompt"], "btag": s["btag"]}
