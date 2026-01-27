## StereoDiffusion inference
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
pip install diffusers==0.35.2 transformers==4.57.1 lightning-utilities==0.15.2 torchmetrics==1.8.2 openexr_numpy
```

### activate both envs
<!-- for galvani only: conda deactivate -->
```
conda activate stereoview
. ./stereoview-venv/bin/activate
cd src/galvani/StereoDiffusion
```

### img2stereo
#### download
```
wget -P midas_models "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt"
sh download_models.sh
```
#### inference
Hyperparameters can be set via `src/galvani/cfg/*_config.json`
```
python3 img2stereo.py
```
