#!/bin/bash

EPOCHS=1
NUM_PROCESSES=4
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
NUM_GENERATIONS=$(($NUM_PROCESSES * $PER_DEVICE_TRAIN_BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))


MODEL_ID="meta-llama/Llama-2-7b-hf"
# MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
export CUDA_VISIBLE_DEVICES=2,3,4,5

accelerate launch --config_file fsdp_config.yaml --num_processes $NUM_PROCESSES \
  scripts/train_grpo.py \
  --model_id $MODEL_ID \
  --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --num_generations $NUM_GENERATIONS \
  --num_train_epochs $EPOCHS \
  --learning_rate 5e-6 \
  --logging_dir ./logs \
  --log_file ./logs/master_train.log