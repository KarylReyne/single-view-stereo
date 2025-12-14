### setup
```
conda activate stereoview
cd src/galvani/StereoDiffusion
```


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
python img2stereo.py --depthmodel_path=midas_models/dpt_hybrid-midas-501f0c75.pt --img_path="../../../data/galvani/fixed_baselines/000/000_left.png" --meta_path="../../../data/galvani/fixed_baselines/000/000_meta.json" --output_prefix="outputs/img2stereo/fixed_baselines-000/out" --baseline_prompt="set B to 0.05"
```
##### complex objects

```
python img2stereo.py --depthmodel_path=midas_models/dpt_hybrid-midas-501f0c75.pt  --img_path="../../../data/galvani/image_collection/Dino/Camera_011/left.jpg" --output_prefix="outputs/img2stereo/Dino-Camera_011/out" --baseline_prompt="set B to 0.05"
```


### prompt2stereo
#### download
```
wget -P midas_models "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt"
wget -P models "https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-2-1/resolve/master/v2-1_768-ema-pruned.ckpt"
```
#### inference
```
python txt2stereo.py --prompt="a cube intersected by a torus" --outfile="cube-vs-torus.png" --depthmodel_path=midas_models/dpt_hybrid-midas-501f0c75.pt --ckpt=models/v2-1_768-ema-pruned.ckpt --config=StableDiffusion/configs/stable-diffusion/v2-inference-v.yaml
```


### depth2stereo
#### download
```
wget -P models "https://www.modelscope.cn/models/stabilityai/stable-diffusion-2-depth/resolve/master/512-depth-ema.ckpt"
```
#### inference
```
python depth2stereoimg.py --ckpt=models/512-depth-ema.ckpt --prompt="a cube intersected by a torus" --init_img="../../../data/galvani/fixed_baselines/000/000_left.png" --depthmodel_path=midas_models/dpt_hybrid-midas-501f0c75.pt --config=StableDiffusion/configs/stable-diffusion/v2-midas-inference.yaml
```
