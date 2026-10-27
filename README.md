<h1 align="left">Adaptive Bayesian Policy Optimization for Post-Training Reasoning Models</h1>

## Installation

You can follow the instruction from [VeRL](https://verl.readthedocs.io/en/latest/start/install.html)

or simply follow what we do:

```
# create env
conda create -n verl python==3.11
conda activate verl

# Make sure you have activated verl conda env
# If you need to run with megatron
bash scripts/install_vllm_sglang_mcore.sh
# Or if you simply need to run with FSDP
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# Install
pip install -e .[gpu,math,vllm]
```

## Training

Train the ABPO with script:
```
bash src/examples/run_qwen25vl7b_geo3k_abpo.sh
```
