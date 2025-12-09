import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

class GeometricEvaluator:
  def compute_flow(self, img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow
  
  def flow_epe(self, f1, f2):
      diff = f1 - f2
      mag = np.sqrt(diff[...,0]**2 + diff[...,1]**2)
      return np.mean(mag)
