# Finetuning via GRPO

## Overview
This project demonstrates fine-tuning the Llama2-7B model using the GRPO algorithm.  
It leverages FSDP distributed training and Accelerate to overcome GPU memory limitations.

## Requirements
- Python 3.10 or higher  
- CUDA 11.8 or higher (tested on A6000)  
- PyTorch 2.1.x  
- Hugging Face Transformers 4.37.x  
- TRL 0.7.x  
- Accelerate 0.28.x
- GPU: A6000 x 4 (ours)

## Installation
```bash
# Create and activate virtual environment (e.g., conda)
conda create -n grpo-env python=3.10 -y
conda activate grpo-env

# Install PyTorch, Transformers, TRL, Accelerate, bitsandbytes
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.37.2 trl==0.7.11 accelerate==0.28.0 bitsandbytes==0.41.1
```

## Accelerate Configuration
```bash
accelerate config --config_file fsdp_config.yaml
```
Example `fsdp_config.yaml`:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
    fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
    fsdp_sharding_strategy: FULL_SHARD
    fsdp_min_num_params: 100000000
    fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
    fsdp_offload_params: false
    fsdp_state_dict_type: SHARDED_STATE_DICT
machine_rank: 0
```

## Training Execution
```bash
# Or execute the provided bash script
bash scripts/run_train.sh
```

## TensorBoard Monitoring
```bash
# Run on server
tensorboard --logdir ./logs --port 6006

# SSH tunneling from local machine
ssh -L 6006:localhost:6006 user@remote.server.address
# Open in local browser
http://localhost:6006/
```

## Results and Logs
- `results/` folder: Checkpoints and final model
- `logs/` folder: TensorBoard event files and training logs

## References
- GRPO original paper: [https://arxiv.org/pdf/2402.03300](https://arxiv.org/pdf/2402.03300)
- Hugging Face TRL: https://github.com/huggingface/trl
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html

