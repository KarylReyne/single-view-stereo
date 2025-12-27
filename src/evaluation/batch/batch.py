import os
from evaluation.evaluation import StereoEvaluator
from tqdm import tqdm

"""
This class is currently expects the following folder structure

gt/
   000/
      000_left.png
      000_right.png

gen/
   000/
      000_gen.png
    
as seen in the current training data.
"""
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

    for folder_name in tqdm(folder_list, desc="Evaluating stereo pairs", unit="scene"):
      gt_subfolder = os.path.join(self.gt_root_folder, folder_name)
      gen_subfolder = os.path.join(self.gen_root_folder, folder_name)

      if not os.path.isdir(gt_subfolder):
        print(f"Skipping {folder_name}: Missing required ground truth subfolder.")
        continue

      if not os.path.isdir(gen_subfolder):
        print(f"Skipping {folder_name}: Missing required generation subfolder.")
        continue
      
      # this could become irrelevant if the images are named 000/left.png, 000/right.png and 000/gen.png
      # in this current version we implement 000/000_left.png, 000/000_right.png and 000/000_gen.png
      """gt_left_image_name = folder_name + "_" + self.left_name_suffix
      gt_right_image_name = folder_name + "_" + self.right_name_suffix
      gen_image_name = folder_name + "_" + self.gen_name_suffix

      gt_left_path = os.path.join(gt_subfolder, gt_left_image_name) 
      gt_right_path = os.path.join(gt_subfolder, gt_right_image_name) 
      gen_image_path = os.path.join(gen_subfolder, gen_image_name)"""

      gt_left_path = os.path.join(gt_subfolder, self.left_name_suffix)
      gt_right_path = os.path.join(gt_subfolder, self.right_name_suffix)
      gen_image_path = os.path.join(gen_subfolder, self.gen_name_suffix)
      meta_path = os.path.join(gt_subfolder, self.meta_name_suffix)

      if not (os.path.exists(gt_left_path) and os.path.exists(gt_right_path) and os.path.exists(gen_image_path) and os.path.exists(meta_path)):
        print(f"Skipping {folder_name}: Missing required images")
        continue 

      evaluator = StereoEvaluator(gt_left_path, gt_right_path, gen_image_path, meta_path)
      evaluator.evaluate()

      metrics = evaluator.perceptual_results | evaluator.geometric_results

      self.results[folder_name] = metrics

    return self.results

      
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