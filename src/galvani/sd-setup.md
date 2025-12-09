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
```
python img2stereo.py --depthmodel_path=midas_models/dpt_hybrid-midas-501f0c75.pt  --img_path="../../../data/galvani/fixed_baselines/000/000_left.png" --baseline_prompt="set B to 6.4"
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
