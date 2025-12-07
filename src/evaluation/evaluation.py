import cv2
from evaluation.geometric.geometric import GeometricEvaluator
from evaluation.perceptual.perceptual import PerceptualEvaluator

class StereoEvaluator:
  def __init__(self, gt_left_path, gt_right_path, gen_right_path, batch_path=None):
    self.gt_left_path = gt_left_path
    self.gt_right_path = gt_right_path
    self.gen_right_path = gen_right_path

    self.batch_path = batch_path

    self.perceptual_evaluator = PerceptualEvaluator()
    self.geometric_evaluator = GeometricEvaluator()

    self.perceptual_results = {}
    self.geometric_results = {}


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

  def evaluate(self):
    gt_left, gt_right, gen_right = self._load_images()

    self.perceptual_results["psnr"] = self.perceptual_evaluator.compute_psnr(gt_right, gen_right)
    self.perceptual_results["ssim"] = self.perceptual_evaluator.compute_ssim(gt_right, gen_right)
    self.perceptual_results["lpips"] = self.perceptual_evaluator.lpips_metric(gt_right, gen_right)

    flow_gt = self.geometric_evaluator.compute_flow(gt_left, gt_right)
    flow_gen = self.geometric_evaluator.compute_flow(gt_left, gen_right)

    self.geometric_results["flow_epe"] = self.geometric_evaluator.flow_epe(flow_gt, flow_gen)



if __name__ == "__main__":
  evaluator = StereoEvaluator(
      gt_left_path="evaluation/000_left.png",
      gt_right_path="evaluation/000_right.png",
      gen_right_path="evaluation/000_gen.png",
  )
   
  evaluator.evaluate()
  results = evaluator.perceptual_results|evaluator.geometric_results
  
  for metric_name, value in results.items():
        print(f"  {metric_name}: {value}")