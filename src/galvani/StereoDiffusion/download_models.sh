mkdir -p models/qwen2.5_7b_instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/qwen2.5_7b_instruct --repo-type model
mkdir -p models/stablediffusion1.4
huggingface-cli download CompVis/stable-diffusion-v1-4 --local-dir models/stablediffusion1.4 --repo-type model