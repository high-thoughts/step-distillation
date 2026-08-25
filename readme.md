# Step Distillation for LLaDA Models

本项目实现了针对 LLaDA (Large Language Adaptive Decoding Architecture) 模型的步骤级蒸馏训练框架，支持多种蒸馏策略，包括 KL 散度蒸馏、特征蒸馏和置信度蒸馏。

## 📋 目录

- [项目概述](#项目概述)
- [主要特性](#主要特性)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [训练方法](#训练方法)
- [评估](#评估)
- [配置说明](#配置说明)
- [相关文档](#相关文档)

## 🎯 项目概述

Step Distillation 是一个专门为 LLaDA 模型设计的知识蒸馏框架。LLaDA 模型采用块级（block-wise）生成策略，本项目通过步骤级蒸馏技术，将教师模型的知识有效地迁移到学生模型中，提升模型在数学推理等任务上的性能。

### 核心创新

- **块级蒸馏**：针对 LLaDA 的块级生成机制设计的蒸馏策略
- **多阈值策略**：教师模型和学生模型使用不同的置信度阈值进行训练
- **多种蒸馏方法**：支持 KL 散度、特征空间、置信度等多种蒸馏方式
- **LoRA 高效微调**：使用 LoRA (Low-Rank Adaptation) 进行参数高效训练

## ✨ 主要特性

### 1. 多种蒸馏策略

- **KL 散度蒸馏** (`train_distillation.py`)
  - 学生模型学习教师模型的输出分布
  - 支持参考模型正则化（可选）
  - 适用于标准的知识蒸馏场景

- **特征蒸馏** (`feature-distillation/train.py`)
  - 在隐藏层空间进行蒸馏
  - 使用 MSE 损失对齐特征表示
  - 更深层次的知识迁移

- **置信度蒸馏** (`feature-distillation/train_confidence.py`)
  - 基于模型预测置信度的蒸馏
  - 动态调整蒸馏强度

- **相似性保持蒸馏** (`feature-distillation/train_sp.py`)
  - Similarity-Preserving (SP) Knowledge Distillation
  - 保持样本间的相似性关系矩阵
  - 结合 KL 散度和相似性矩阵损失
  - 参考论文: arXiv:1907.09682

### 2. 训练优化

- **DeepSpeed 集成**：支持 ZeRO Stage 2 优化，高效利用多卡资源
- **混合精度训练**：支持 BF16/FP16，加速训练并节省显存
- **梯度累积**：支持大批次训练
- **学习率调度**：Cosine 退火 + Warmup
- **Weights & Biases 集成**：完整的训练监控和可视化

### 3. 灵活的数据处理

- 支持 Parquet、JSON、JSONL 等多种数据格式
- 灵活的键名映射（prompt_key, response_key）
- 自动数据集合并和预处理
- GSM8K 数学推理数据集专用加载器

## 📁 项目结构

```
step-distillation/
├── train_distillation.py          # 主训练脚本（KL 散度蒸馏）
├── sft_expand_dataset.py          # 数据集加载和预处理
├── train.sh                       # 训练启动脚本
├── step-distillation.yaml         # 训练配置文件
├── default_config.yaml            # Accelerate 配置文件
├── env.yaml                       # Conda 环境配置文件
│
├── doc/                           # 项目文档目录
│   ├── Step-Distillation 20251106-1113.docx  # 周报 2025/11/06-11/13
│   ├── Step-Distillation 20251113-1120.docx  # 周报 2025/11/13-11/20
│   ├── Step-Distillation 20251120-1127.docx  # 周报 2025/11/20-11/27
│   └── Step-Distillation 20251127-1204.docx  # 周报 2025/11/27-12/04
│
├── feature-distillation/          # 特征蒸馏相关脚本
│   ├── train.py                   # 特征蒸馏训练
│   ├── train_sp.py                # 相似矩阵蒸馏训练
│   ├── train_confidence.py        # 置信度蒸馏训练
│   ├── feature-distillation.yaml  # 特征蒸馏配置
│   └── sft_expand_dataset.py      # 数据集加载器（副本）
│
├── eval/                          # 评估相关脚本
│   ├── eval.py                    # GSM8K 评估脚本
│   ├── gsm8k.py                   # GSM8K 数据集加载
│   ├── run_eval_batch.sh          # 批量评估脚本
│   ├── convert_deepspeed_to_peft.py  # 模型转换工具
│   └── LLaDA-Prometheus/          # LLaDA 模型定义
│
└── datasets/                      # 数据集目录
    └── gsm8k/                     # GSM8K 数据集
```

## 🔧 环境配置

### 系统要求

- **Python**: 3.10.17
- **PyTorch**: 2.7.0
- **CUDA**: 12.x (推荐 CUDA 12.6)
- **操作系统**: Linux (也支持 Windows)
- **推荐**: 多卡 GPU 环境，支持 DeepSpeed

### 方法一：使用 conda 环境文件（推荐）

项目提供了完整的 `env.yaml` 文件，可以一键创建环境：

```bash
# 使用 env.yaml 创建环境
conda env create -f env.yaml

# 激活环境
conda activate step-distillation
```

### 方法二：手动安装

```bash
# 1. 创建 conda 环境
conda create -n step-distillation python=3.10
conda activate step-distillation

# 2. 安装 PyTorch (CUDA 12.x)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装核心依赖
pip install transformers==4.46.2
pip install accelerate==1.6.0
pip install deepspeed==0.18.2
pip install peft==0.15.2
pip install datasets==4.2.0

# 4. 安装其他依赖
pip install pandas pyarrow safetensors
pip install wandb  # 可选，用于训练监控
pip install tqdm pyyaml

# 5. 安装 Flash Attention (可选，加速训练)
pip install flash-attn==2.8.3
```

### 核心依赖版本

| 包名 | 版本 | 说明 |
|------|------|------|
| `torch` | 2.7.0 | PyTorch 深度学习框架 |
| `transformers` | 4.46.2 | HuggingFace Transformers |
| `accelerate` | 1.6.0 | 分布式训练加速 |
| `deepspeed` | 0.18.2 | DeepSpeed 优化器 |
| `peft` | 0.15.2 | LoRA 等参数高效微调 |
| `datasets` | 4.2.0 | 数据集加载 |
| `flash-attn` | 2.8.3 | Flash Attention 加速 |
| `wandb` | 0.22.2 | 训练监控（可选）|

### Accelerate 配置

项目使用 Accelerate 进行分布式训练。配置方式：

**方法一：使用项目提供的配置（推荐）**

```bash
# 项目已包含 default_config.yaml，可直接使用
# 配置说明：
# - 分布式类型: DeepSpeed
# - ZeRO Stage: 2
# - 混合精度: BF16
# - 梯度累积: 8 步
```

**方法二：自定义配置**

```bash
# 交互式生成配置
accelerate config

# 推荐配置选项：
# - Compute environment: LOCAL_MACHINE
# - Distributed type: DEEPSPEED
# - DeepSpeed config: ZeRO Stage 2
# - Mixed precision: bf16
```

**default_config.yaml 配置详情**：
```yaml
distributed_type: DEEPSPEED
deepspeed_config:
  zero_stage: 2                    # ZeRO 优化阶段
  gradient_accumulation_steps: 8   # 梯度累积步数
  gradient_clipping: 1.0           # 梯度裁剪
  offload_optimizer_device: cpu    # 优化器卸载到 CPU
  offload_param_device: cpu        # 参数卸载到 CPU
mixed_precision: bf16              # BF16 混合精度
num_processes: 2                   # GPU 数量（根据实际调整）
```

## 🚀 快速开始

### 1. 准备数据

将您的训练数据准备为 Parquet 格式，确保包含以下字段：
- `question` (或自定义的 prompt_key)
- `answer` (或自定义的 response_key)

```python
import pandas as pd

data = pd.DataFrame([
    {"question": "What is 2+2?", "answer": "2+2=4. The answer is \\boxed{4}."},
    # 更多数据...
])
data.to_parquet("datasets/gsm8k/train.parquet")
```

### 2. 配置训练参数

编辑 `step-distillation.yaml`：

```yaml
# 模型路径
model_path: "/path/to/your/llada/model"
ref_model_path: "/path/to/reference/model"  # 可选

# LoRA 配置
lora_rank: 32
lora_alpha: 64

# 数据配置
train_files: ["datasets/gsm8k/train.parquet"]
val_files: ["datasets/gsm8k/test.parquet"]
max_length: 256

# 蒸馏配置
block_length: 64
threshold_high: 0.9  # 教师模型阈值
threshold_low: 0.6   # 学生模型阈值
beta: 0.1            # 参考模型损失权重

# 训练配置
lr: 1e-5
total_epochs: 1
train_batch_size: 4
save_steps: 32
```

### 3. 启动训练

```bash
# 使用提供的训练脚本
bash train.sh

# 或直接使用 accelerate
accelerate launch train_distillation.py --config_file step-distillation.yaml
```

### 4. 监控训练

如果启用了 W&B：

```yaml
use_wandb: true
wandb_project: "step-distillation"
wandb_run_name: "experiment_name"
```

访问 [https://wandb.ai](https://wandb.ai) 查看训练进度。

## 📚 训练方法

### KL 散度蒸馏

最标准的蒸馏方法，学生模型学习教师模型的输出分布。

```bash
accelerate launch train_distillation.py --config_file step-distillation.yaml
```

**损失函数**：
```
Loss = KL(Student || Teacher) + β * KL(Teacher || Reference)
```

### 特征蒸馏

在隐藏层空间进行蒸馏，捕获更深层次的知识。

```bash
cd feature-distillation
accelerate launch train.py --config_file feature-distillation.yaml
```

**损失函数**：
```
Loss = MSE(H_student, H_teacher)
```

其中 H 表示最后一层的隐藏状态。

### 置信度蒸馏

结合特征蒸馏和置信度感知损失，鼓励学生模型在较低阈值下也能产生高置信度预测。

```bash
cd feature-distillation
accelerate launch train_confidence.py --config_file feature-distillation.yaml
```

**损失函数**：
```
Loss = L_feature + λ * L_confidence

其中:
L_feature = MSE(H_student, H_teacher)
L_confidence = MSE(max(P_student), max(P_teacher))
P = softmax(logits)  # 预测概率分布
```

**关键配置**：
```yaml
use_confidence_loss: true        # 启用置信度损失
confidence_loss_weight: 0.3      # 置信度损失权重
```

**优势**：帮助学生模型在使用较低阈值时仍能保持高质量的预测置信度。

### 相似性保持蒸馏 (Similarity-Preserving)

通过保持样本间的相似性关系进行知识迁移，结合 KL 散度损失。

```bash
cd feature-distillation
accelerate launch train_sp.py --config_file feature-distillation.yaml
```

**损失函数**：
```
Loss = α * L_SP + β * L_KL

其中:
L_SP = ||G_teacher - G_student||²_F / B²
G = normalize(Q * Q^T)  # 样本间相似性矩阵
L_KL = KL(Student || Teacher)
```

**关键配置**：
```yaml
sp_weight: 1.0          # 相似性损失权重
kl_weight: 1.0          # KL 散度损失权重
kl_temperature: 1.0     # KL 散度温度参数
```

**注意**：SP 损失需要 batch_size > 1 才能计算样本间关系。

## 📊 评估

### GSM8K 评估

评估模型在 GSM8K 数学推理任务上的性能：

```bash
cd eval

# 评估单个 checkpoint
python eval.py \
    --base_model_path /path/to/base/model \
    --checkpoint_path /path/to/checkpoint \
    --batch_size 8 \
    --max_gen_length 256 \
    --block_length 64 \
    --threshold 0.9

# 评估基础模型（不加载 adapter）
python eval.py \
    --base_model_path /path/to/base/model \
    --base_only \
    --batch_size 8

# 批量评估多个 checkpoints
bash run_eval_batch.sh
```

### 评估指标

- **准确率 (Accuracy)**：正确回答的问题比例
- **平均解码迭代次数**：每个样本的平均块级解码迭代次数
- **详细结果**：每个样本的预测和正确答案对比

评估结果保存为：
- `gsm8k_eval_*.jsonl`：每个样本的详细结果
- `accuracy_metrics.json`：汇总的准确率指标

## ⚙️ 配置说明

### 核心配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model_path` | 基础模型路径 | 必需 |
| `ref_model_path` | 参考模型路径（可选） | null |
| `lora_rank` | LoRA 秩 | 32 |
| `lora_alpha` | LoRA alpha 参数 | 64 |
| `block_length` | 块长度 | 64 |
| `threshold_high` | 教师模型置信度阈值 | 0.9 |
| `threshold_low` | 学生模型置信度阈值 | 0.6 |
| `beta` | 参考模型损失权重 | 0.1 |
| `lr` | 学习率 | 1e-5 |
| `train_batch_size` | 训练批次大小 | 4 |
| `gradient_accumulation_steps` | 梯度累积步数 | 1 |
| `max_length` | 最大序列长度 | 256 |
| `save_steps` | 保存间隔（步数） | 32 |
| `mixed_precision` | 混合精度类型 | 'bf16' |

### LoRA 目标模块

默认对以下模块应用 LoRA：
- `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `gate_proj`, `up_proj`, `down_proj`

### 数据配置

```yaml
train_files: 
  - "datasets/gsm8k/train.parquet"
  - "datasets/custom/*.parquet"  # 支持通配符

prompt_key: "question"  # 数据集中问题字段名
response_key: "answer"  # 数据集中答案字段名
```

## 📖 相关文档

### 本地文档

项目 `doc/` 目录包含详细的周报文档，记录了项目的开发进展、实验结果和技术细节：

- **Step-Distillation 20251106-1113.docx** - 第一周周报（2025/11/06-11/13）
  - 项目启动和初始设计
  - 基础框架搭建
  
- **Step-Distillation 20251113-1120.docx** - 第二周周报（2025/11/13-11/20）
  - KL 散度蒸馏实现
  - 初步实验结果
  
- **Step-Distillation 20251120-1127.docx** - 第三周周报（2025/11/20-11/27）
  - 特征蒸馏和置信度蒸馏
  - 性能优化
  
- **Step-Distillation 20251127-1204.docx** - 第四周周报（2025/11/27-12/04）
  - 相似性保持蒸馏
  - 综合评估和总结

### 在线文档

详细设计文档和实验结果也可参考以下飞书文档：

1. **项目总览**  
   [https://ai.feishu.cn/wiki/A6BIwiM2HiQ9FhkXxQbcE83nnGg](https://ai.feishu.cn/wiki/A6BIwiM2HiQ9FhkXxQbcE83nnGg)

2. **蒸馏方法详解**  
   [https://ai.feishu.cn/wiki/IauTw8SO7iklwoklsvSc2gIOnW6](https://ai.feishu.cn/wiki/IauTw8SO7iklwoklsvSc2gIOnW6)

3. **实验结果与分析**  
   [https://ai.feishu.cn/wiki/P8q9wYI2YihYd1kwxkUcuplgnPQ](https://ai.feishu.cn/wiki/P8q9wYI2YihYd1kwxkUcuplgnPQ)

4. **最佳实践指南**  
   [https://ai.feishu.cn/wiki/QayHwdjkpisrLokkqAicyQu1n6g](https://ai.feishu.cn/wiki/QayHwdjkpisrLokkqAicyQu1n6g)

## 🔍 常见问题

### Q: 如何选择合适的阈值？

- `threshold_high`（教师模型）：通常设置为 0.9，确保高质量的生成
- `threshold_low`（学生模型）：通常设置为 0.6-0.7，允许更多探索

### Q: 如何恢复训练？

```yaml
resume_from_checkpoint: "./output/checkpoint-1000"
# 或自动恢复最新的 checkpoint
resume_from_checkpoint: "latest"
```

## 📝 输出文件

训练过程中会生成以下文件：

```
output/
└── distillation_20231215_143022/
    ├── checkpoint-32/
    │   ├── adapter_model.safetensors  # LoRA 权重
    │   ├── adapter_config.json        # LoRA 配置
    │   ├── training_state.pth         # 训练状态
    │   ├── sample_output.json         # 推理样本
    │   └── README.md                  # Checkpoint 说明
    ├── checkpoint-64/
    └── last_checkpoint.txt            # 最新 checkpoint 路径
```

**Happy Training! 🎉**