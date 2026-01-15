## StereoDiffusion inference
<!--
Due to numerous dependency issues, I have to use both a conda and a venv environment. The conda env only contains python 3.10 and pip, all packages are installed in the venv. To install the packages I no longer use a single yml file but instead install from each of the cloned repos' individual requirements.txt files. Refer to the setup instructions below for how I got every up and running (after a LOT of trial-and-error...). 
-->
### new conda env install
```
conda create --name stereoview python=3.10
conda install pip
```
### new venv install
```
python3 -m venv ./stereoview-venv
. ./stereoview-venv/bin/activate
```
IMPORTANT: if not already present, install torch with cuda support
see https://pytorch.org/get-started/locally/ for details
```
cd src/galvani/StereoDiffusion/StableDiffusion
pip install -r requirements.txt
cd ../PromptToPrompt
pip install -r requirements.txt
cd ..
pip install timm "numpy<2" tueplots accelerate ijson
pip install diffusers==0.35.2 transformers==4.57.1
```

### activation chain for both envs
<!-- for galvani only: conda deactivate -->
```
conda activate stereoview
. ./stereoview-venv/bin/activate
cd src/galvani/StereoDiffusion
```
<!-- now you're ready to run img2stereo.py ;) -->


### img2stereo
#### download
```
wget -P midas_models "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt"
sh download_model.sh
```
#### inference
 - custom baseline can be set with `--baseline_prompt` (requires focal length in metadata (for now) -> does not work for the primitives dataset!)
 - whether DPT should estimate disparity or depth can be controlled by passing `--estimate_only_depth` (I think... ;) the option just sets DPTDepthModel.invert = True)
##### primitives
```
python img2stereo.py --depthmodel_path=midas_models/dpt_hybrid-midas-501f0c75.pt --img_path="../../../data/galvani/fixed_baselines/000/000_left.png" --meta_path="../../../data/galvani/fixed_baselines/000/000_meta.json" --output_prefix="outputs/img2stereo/fixed_baselines-000/out" --baseline_prompt="set B to 0.05 and f to 40.2"
```
##### complex objects

```
python img2stereo.py --depthmodel_path=midas_models/dpt_hybrid-midas-501f0c75.pt  --img_path="../../../data/galvani/image_collection/Dino/Camera_011/left.jpg" --output_prefix="outputs/img2stereo/Dino-Camera_011/out" --baseline_prompt="set B to 0.05 and f to 40.2"
```

<!-- 
python3 img2stereo.py --depthmodel_path=midas_models/dpt_hybrid-midas-501f0c75.pt  --img_path="../../../resources/cat_left_gt.png" --output_prefix="../../../resources/stereodiffusion/cat" --baseline_prompt="set B to 0.12 and f to 39.1" --estimate_only_depth

python3 img2stereo.py --depthmodel_path=midas_models/dpt_hybrid-midas-501f0c75.pt  --img_path="../../../resources/mona_lisa_left_gt.png" --output_prefix="../../../resources/stereodiffusion/mona_lisa" --baseline_prompt="set B to 0.12 and f to 39.1" --estimate_only_depth

python3 img2stereo.py --depthmodel_path=midas_models/dpt_hybrid-midas-501f0c75.pt  --img_path="../../../resources/car_left.jpg" --output_prefix="../../../resources/stereodiffusion/car" --meta_path="../../../resources/car_meta.json" --estimate_only_depth 
-->

<!--
python3 img2stereo_train.py --depthmodel_path=midas_models/dpt_hybrid-midas-501f0c75.pt  --img_path="../../../resources/cat_left_gt.png" --output_prefix="../../../resources/stereodiffusion/TRAIN-TEST_cat" --baseline_prompt="set B to 0.12 and f to 39.1"
-->


## zero123 training - DEPRECATED
### for galvani only
```
srun --job-name "train01" --partition=a100-galvani --ntasks=1 --nodes=1 --gres=gpu:4 --time 1:00:00 --pty bash
```
### setup
```
conda activate stereoview-train 
cd src/galvani/zero_123/zero123
```
### download/extract
```
wget -P models "https://cv.cs.columbia.edu/zero123/assets/sd-image-conditioned-v2.ckpt"
unzip ./valid_paths.json.zip -d ./view_release
```
### additional dependencies
```
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
```
### train
```
python main.py \
    -t \
    --base configs/sd-objaverse-finetune-c_concat-256.yaml \
    --gpus 0,1,2,3 \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from models/sd-image-conditioned-v2.ckpt
```