import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

class GeometricEvaluator:
  def flow_metric(self, img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow
  
  
  def flow_error(self, f1, f2, baseline):
    diff = f1 - f2
    mag = np.sqrt(diff[..., 0]**2 + diff[..., 1]**2)
    return np.mean(mag) / baseline 

  def compute_disparity(self, left, right):
    g1 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,   # durch 16 teilbar
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2
    )
    disp = stereo.compute(g1, g2).astype(np.float32) / 16.0
    return disp
  
  def disparity_error(self, disp_gt, disp_gen, baseline, mask=None):
    if mask is None:
        mask = np.isfinite(disp_gt) & np.isfinite(disp_gen)
    return np.mean(np.abs(disp_gt[mask] - disp_gen[mask])) / baseline


  def bad_pixel_rate(self, d_gt, d_gen, thresh=1.0):
    mask = np.isfinite(d_gt)
    return np.mean(np.abs(d_gt[mask] - d_gen[mask]) > thresh)
