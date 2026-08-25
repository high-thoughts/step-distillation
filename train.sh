#!/bin/bash

# --- 训练脚本配置 ---

# 1. 立即退出：如果任何命令失败，脚本将停止。
set -e

# 2. 变量：在这里设置您的文件名
# 您的 conda 环境名称
ENV_NAME="step-distillation"

# 您的主 Python 训练脚本
TRAIN_SCRIPT="train_distillation.py"

# 您的训练超参数 YAML 文件 (lr, model_path, lora_rank, etc.)
TRAIN_CONFIG="step-distillation.yaml"

# 启动训练
# 'accelerate launch' 会自动读取您的 accelerate_config.yaml (环境配置)
# 我们使用 '--config_file' 将超参数配置传递给 Python 脚本
accelerate launch ${TRAIN_SCRIPT} --config_file ${TRAIN_CONFIG}

echo ""
echo "--- 训练完成 ---"