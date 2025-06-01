# train_grpo_accelerate.py
import argparse
import logging
import os

from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOTrainer, GRPOConfig
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description="Train Llama2-7B with GRPO via Accelerate+FSDP")
    parser.add_argument("--model_id", type=str, required=True,
                        help="HuggingFace 모델 경로 (예: meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="모델과 로그가 저장될 디렉터리")
    parser.add_argument("--logging_dir", type=str, default="./logs",
                        help="로그 파일 및 TensorBoard 디렉터리")
    parser.add_argument("--log_file", type=str, default=None,
                        help="마스터 노드가 기록할 로그 파일 경로")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_generations", type=int, default=2,
                        help="GRPO 그룹 내 생성 샘플 수 (num_generations or group_size)")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="몇 step마다 로그를 기록할지")
    return parser.parse_args()

# Logging 콜백 정의
class TensorBoardCallback(TrainerCallback):
    def __init__(self, log_dir):
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

    def on_step_end(self, args, state, control, **kwargs):
        rank = int(os.environ.get("RANK", 0))
        if rank != 0:
            return
        if state.log_history:
            last = state.log_history[-1]
            logging.info(f"Step {state.global_step} | Loss: {last.get('loss','n/a')} | Learning Rate: {last.get('learning_rate','n/a')} | Batch Size: {args.per_device_train_batch_size}")
            logging.info(f"Current Batch: {last.get('current_batch','n/a')} | Group Size: {args.num_generations}")
            if self.writer is not None:
                for key, value in last.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(key, value, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer is not None:
            self.writer.close()
            
# GRPO 보상 함수 정의 TODO: 실제 작업에 맞는 보상 함수를 구현해야 합니다.
def reward_len(completions, **kwargs):
    return [-abs(20 - len(c)) for c in completions]

def main():
    args = parse_args()

    # -- logging setup (마스터 프로세스만 파일로 기록)
    os.makedirs(args.logging_dir, exist_ok=True)
    log_file = args.log_file or os.path.join(args.logging_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file) if int(os.environ.get("RANK", 0)) == 0 else logging.NullHandler(),
            logging.StreamHandler()
        ]
    )

    # 1) 데이터셋 로드
    train_dataset = load_dataset("trl-lib/tldr", split="train")

    # 2) 토크나이저만 미리 로드 (모델 로드는 GRPOTrainer가 처리)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # 3) GRPO 설정 (num_generations = group_size)
    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        num_generations=args.num_generations,
        fp16=True,
        remove_unused_columns=False,
    )

    # 4) GRPOTrainer 초기화
    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        args=grpo_args,
        train_dataset=train_dataset,
        reward_funcs=reward_len,
        callbacks=[TensorBoardCallback(args.logging_dir)],
    )

    # 5) 학습 시작
    trainer.train()

if __name__ == "__main__":
    main()