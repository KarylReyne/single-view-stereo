import cv2
from evaluation.geometric.geometric import GeometricEvaluator
from evaluation.perceptual.perceptual import PerceptualEvaluator
import json

class StereoEvaluator:
  def __init__(self, gt_left_path, gt_right_path, gen_right_path, meta_path, batch_path=None):
    self.gt_left_path = gt_left_path
    self.gt_right_path = gt_right_path
    self.gen_right_path = gen_right_path
    self.meta_path = meta_path
    self.batch_path = batch_path

    self.perceptual_evaluator = PerceptualEvaluator()
    self.geometric_evaluator = GeometricEvaluator()

    self.perceptual_results = {}
    self.geometric_results = {}

    self.meta = self._load_meta()


  def _load_images(self):
    gt_left = cv2.imread(self.gt_left_path, cv2.IMREAD_COLOR)
    gt_right = cv2.imread(self.gt_right_path, cv2.IMREAD_COLOR)
    gen_right = cv2.imread(self.gen_right_path, cv2.IMREAD_COLOR)


    if gt_left is None:
        raise FileNotFoundError(f"Could not read gt_left image at {self.gt_left_path}")
    if gt_right is None:
        raise FileNotFoundError(f"Could not read gt_right image at {self.gt_right_path}")
    if gen_right is None:
        raise FileNotFoundError(f"Could not read gen_right image at {self.gen_right_path}")
    
    h, w = gt_right.shape[:2]
    if gen_right.shape[:2] != (h, w):
        print("Size of generated image does not match ground truth. Resizing.")
        gen_right = cv2.resize(gen_right, (w, h), interpolation=cv2.INTER_AREA)
    
    return gt_left, gt_right, gen_right


  def _load_meta(self):
    with open(self.meta_path, "r") as f:
        meta = json.load(f)
    return {
        "baseline": meta["baseline_m"],
        "fov": meta["fov_deg"],
    }


  def evaluate(self):
    gt_left, gt_right, gen_right = self._load_images()
    baseline = self.meta["baseline"]

    self.perceptual_results["psnr"] = self.perceptual_evaluator.psnr_metric(gt_right, gen_right)
    self.perceptual_results["ssim"] = self.perceptual_evaluator.ssim_metric(gt_right, gen_right)
    self.perceptual_results["lpips"] = self.perceptual_evaluator.lpips_metric(gt_right, gen_right)

    # normalized optical flow error
    flow_gt = self.geometric_evaluator.flow_metric(gt_left, gt_right)
    flow_gen = self.geometric_evaluator.flow_metric(gt_left, gen_right)

    self.geometric_results["optical_flow_error"] = self.geometric_evaluator.flow_error(flow_gt, flow_gen, baseline)

    # normalized disparity error
    disp_gt  = self.geometric_evaluator.compute_disparity(gt_left, gt_right)
    disp_gen = self.geometric_evaluator.compute_disparity(gt_left, gen_right)

    baseline = self.meta["baseline"]
    self.geometric_results["disp_epe"] = self.geometric_evaluator.disparity_error(disp_gt, disp_gen, baseline)