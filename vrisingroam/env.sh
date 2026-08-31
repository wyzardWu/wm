# source this file: `. env.sh`
export ROOT=/home/yuzhewu/vrisingroam
export PYTHONPATH=$ROOT/ReactiveGWM/DiffSynth-Studio:$ROOT/ReactiveGWM:$ROOT:${PYTHONPATH:-}
export PATH=/home/yuzhewu/miniconda3/envs/rgwm/bin:$PATH
export BASE_MODEL=/nfs/zeqingwang/models/base_model
export DIT_JSON='[["'$BASE_MODEL'/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors","'$BASE_MODEL'/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors","'$BASE_MODEL'/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"],"'$BASE_MODEL'/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth","'$BASE_MODEL'/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"]'
export TOKENIZER=$BASE_MODEL/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl
