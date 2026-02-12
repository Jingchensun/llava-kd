export WANDB_API_KEY="595cc8071abc681aa346ae6017f73fc16a9b2033"  # 替换为你的API Key
export WANDB_MODE=online  # 确保 wandb 处于在线模式

bash 1_train_qwen2_distill.sh
bash 2_train_sft_student_only.sh
bash 3_train_qwen2_distill_after_sft.sh
