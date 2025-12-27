import os
from evaluation.evaluation import StereoEvaluator, PerceptualEvaluator, GeometricEvaluator
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json

class BatchHandling:
  def __init__(self, gt_root_folder, gen_root_folder, left_name_suffix = "left.jpg", right_name_suffix = "right.jpg", gen_name_suffix = "gen.jpg", meta_name_suffix = "meta.json"):
    self.gt_root_folder = gt_root_folder
    self.gen_root_folder = gen_root_folder

    self.left_name_suffix = left_name_suffix
    self.right_name_suffix = right_name_suffix
    self.gen_name_suffix = gen_name_suffix
    self.meta_name_suffix = meta_name_suffix

    self.results = {}


  def evaluate_batch(self):
    folder_list = sorted(os.listdir(self.gt_root_folder))
    perceptual_evaluator = PerceptualEvaluator()
    geometric_evaluator = GeometricEvaluator()

    for folder_name in tqdm(folder_list, desc="Evaluating stereo pairs", unit="scene"):
      gt_subfolder = os.path.join(self.gt_root_folder, folder_name)
      gen_subfolder = os.path.join(self.gen_root_folder, folder_name)

      if not os.path.isdir(gt_subfolder):
        print(f"Skipping {folder_name}: Missing required ground truth subfolder.")
        continue

      if not os.path.isdir(gen_subfolder):
        print(f"Skipping {folder_name}: Missing required generation subfolder.")
        continue
      
      gt_left_path = os.path.join(gt_subfolder, self.left_name_suffix)
      gt_right_path = os.path.join(gt_subfolder, self.right_name_suffix)
      gen_image_path = os.path.join(gen_subfolder, self.gen_name_suffix)
      meta_path = os.path.join(gt_subfolder, self.meta_name_suffix)

      if not (os.path.exists(gt_left_path) and os.path.exists(gt_right_path) and os.path.exists(gen_image_path) and os.path.exists(meta_path)):
        print(f"Skipping {folder_name}: Missing required images")
        continue 

      evaluator = StereoEvaluator(gt_left_path=gt_left_path, gt_right_path=gt_right_path, gen_right_path=gen_image_path, meta_path=meta_path, perceptual_evaluator=perceptual_evaluator, geometric_evaluator=geometric_evaluator)
      
      evaluator.evaluate()

      metrics = evaluator.perceptual_results | evaluator.geometric_results

      self.results[folder_name] = metrics

    return self.results

  def summarize_all(self):
    if not self.results:
        return {}

    metric_names = list(next(iter(self.results.values())).keys())
    summary = {}

    for key in metric_names:
        vals = [v[key] for v in self.results.values() if key in v]
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals))
        }

    return summary
  
  def export_json(self, out_path="export.json"):
    payload = {
        "created_at": datetime.now().isoformat(),
        "num_scenes": len(self.results),
        "per_scene": self.results,
        "summary": self.summarize_all()
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
  evaluator = BatchHandling(
      gt_root_folder="../data/galvani/image_collection/Car",
      gen_root_folder="../data/galvani/image_collection/Car",
      gen_name_suffix = "right.jpg"
  )
   
  evaluator.evaluate_batch()
  results = evaluator.results
  
  for scene, metrics in results.items():
      print(f"\nScene {scene}")
      for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value}")

  # mean + std sumary metrics
  summary = evaluator.summarize_all()
  for metric_name, stats in summary.items():
    print(f"\nMetric {metric_name}")
    for stat_name, value in stats.items():
      print(f"  {stat_name}: {value}")


  evaluator.export_json()