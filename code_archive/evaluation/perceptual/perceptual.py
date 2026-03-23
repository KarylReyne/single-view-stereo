import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import torch
import lpips
import cv2

class PerceptualEvaluator:
  def __init__(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)
    
  # Peak signal-to-noise ratio
  def psnr_metric(self, img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

  # Structural similarity index measure
  def ssim_metric(self, img1, img2):
    return ssim(img1, img2, data_range=255, channel_axis=-1)
  
  # Learned Perceptual Image Patch Similarity
  def lpips_metric(self, img1, img2):
    # Convert BGR to RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    t1 = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    t2 = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # normalise to [-1,1]
    t1 = t1 * 2 - 1  
    t2 = t2 * 2 - 1

    t1 = t1.to(self.device)
    t2 = t2.to(self.device)

    return self.lpips_model(t1, t2).item()