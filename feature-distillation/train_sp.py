import os
import torch
# import torch_npu
import torch.nn.functional as F
from torch import optim 
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from sft_expand_dataset import SFTExpandDataset
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import get_cosine_schedule_with_warmup
from safetensors.torch import save_file as safe_save_file
import yaml
import argparse
import types 
from datetime import datetime
import math
import json

def setup_tokenizer(config):
    """加载 Tokenizer 并设置 pad_token。"""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer pad_token set to eos_token.")
    return tokenizer

def setup_models(config, accelerator):
    """加载主模型 (student)。"""
    
    torch_dtype = torch.bfloat16 if config.mixed_precision == 'bf16' else torch.float16
    if accelerator.is_main_process:
        print(f"Using torch_dtype: {torch_dtype} based on config")
    
    if accelerator.is_main_process:
        print("Loading student model...")
    model_config = AutoConfig.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
    )
    model_config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        config=model_config,
        torch_dtype=torch_dtype, # <-- 应用动态 dtype
        attn_implementation="eager",
        trust_remote_code=config.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    if config.lora_rank > 0:
        if accelerator.is_main_process:
            print(f"Applying LoRA with rank {config.lora_rank}...")
        
        # 为自定义模型添加 prepare_inputs_for_generation 方法(如果不存在)
        if not hasattr(model, 'prepare_inputs_for_generation'):
            def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
                if past_key_values is not None:
                    input_ids = input_ids[:, -1:]
                return {
                    "input_ids": input_ids,
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache", True),
                }
            model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(model, model.__class__)
        
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        
        if accelerator.is_main_process:
            model.print_trainable_parameters()
    
    return model

def setup_dataloaders(config, tokenizer, accelerator):

    train_dataset = SFTExpandDataset(
        parquet_files=config.train_files,
        tokenizer=tokenizer,
        prompt_key=config.prompt_key,
        response_key=config.response_key,
        max_length=config.max_length,
        truncation=config.truncation,
    )
    if accelerator.is_main_process:
        print(f"Train dataset size (total): {len(train_dataset)}")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        collate_fn=SFTExpandDataset.collate_fn,
    )
    return train_dataloader

def save_checkpoint(accelerator, output_dir, global_step, epoch, prefix="step", tokenizer=None, config=None, run_dir=None):
    base_dir = run_dir if run_dir else output_dir
    ckpt_dir = os.path.join(base_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    unwrapped_model = accelerator.unwrap_model(accelerator._models[0])
    
    # 保存 LoRA 权重
    if hasattr(unwrapped_model, 'save_pretrained'):
        unwrapped_model.save_pretrained(ckpt_dir, safe_serialization=True)
    else:
        safe_save_file(unwrapped_model.state_dict(), os.path.join(ckpt_dir, "adapter_model.safetensors"))
    
    # 保存 tokenizer
    if tokenizer and accelerator.is_main_process:
        try:
            tokenizer.save_pretrained(ckpt_dir)
            print(f"Tokenizer saved to {ckpt_dir}")
        except Exception as e:
            print(f"Failed to save tokenizer: {e}")
    
    # 保存 README.md
    if accelerator.is_main_process and config:
        try:
            readme = f"""---
library_name: peft
base_model: {getattr(config, 'model_path', 'unknown')}
---

# LoRA Adapter - Checkpoint {global_step}

This is a LoRA adapter trained using step distillation.

## Training Info
- **Global Step**: {global_step}
- **Epoch**: {epoch + 1}
- **LoRA Rank**: {getattr(config, 'lora_rank', 'unknown')}
- **LoRA Alpha**: {getattr(config, 'lora_alpha', 'unknown')}
- **Target Modules**: {getattr(config, 'target_modules', 'unknown')}

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("{getattr(config, 'model_path', 'base_model_path')}")
tokenizer = AutoTokenizer.from_pretrained("{ckpt_dir}")
model = PeftModel.from_pretrained(base_model, "{ckpt_dir}")

inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs)
```
"""
            open(os.path.join(ckpt_dir, "README.md"), "w", encoding="utf-8").write(readme)
        except Exception as e:
            print(f"Failed to save README.md: {e}")
    
    # 保存训练状态
    try:
        torch.save({"global_step": int(global_step), "epoch": int(epoch)}, 
                   os.path.join(ckpt_dir, "training_state.pth"))
    except Exception as e:
        print(f"Failed to save training_state.pth: {e}")
    
    # 保存最后的 checkpoint 路径
    if accelerator.is_main_process:
        try:
            open(os.path.join(base_dir, "last_checkpoint.txt"), "w").write(ckpt_dir)
        except Exception as e:
            print(f"Failed to write last_checkpoint.txt: {e}")
    
    return ckpt_dir

def _build_math_prompt(tokenizer):
    msg_text = "Please calculate 17 + 25, and provide the reasoning process and the final answer."
    messages = [{"role": "user", "content": msg_text}]
    try:
        prompt = (tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
                  if hasattr(tokenizer, "apply_chat_template") 
                  else f"User: {msg_text}\nAssistant:")
    except Exception:
        prompt = msg_text
    return messages, prompt

def evaluate_and_save_sample(accelerator, tokenizer, model, ckpt_dir, config, global_step, epoch):
    if not accelerator.is_main_process:
        return None
    try:
        model.eval()
        messages, prompt = _build_math_prompt(tokenizer)
        inputs = {k: v.to(model.device) for k, v in tokenizer(prompt, return_tensors="pt").items()}
        
        gen_kwargs = {
            **inputs,
            "attention_mask": inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
            "max_gen_length": int(getattr(config, "infer_max_length", getattr(config, "max_gen_length", 1024))),
            "block_length": int(getattr(config, "block_length", 64)),
            "threshold": float(getattr(config, "threshold_high", 0.9)),
            "eos_token_id": int(getattr(config, "infer_eos_token_id", getattr(config, "train_eos_token_id", 126348))),
        }
        
        with torch.no_grad():
            result = model.generate(**gen_kwargs)
        
        text = tokenizer.batch_decode(result, skip_special_tokens=True)
        payload = {
            "prompt": messages[0]["content"],
            "output": text[0] if isinstance(text, list) and text else str(text),
            "step": int(global_step),
            "epoch": int(epoch + 1),
            "time": datetime.now().isoformat(timespec="seconds"),
            "model_path": str(getattr(config, "model_path", "")),
            "has_lora": bool(getattr(config, "lora_rank", 0) > 0),
            "ckpt_dir": ckpt_dir,
        }
        
        out_path = os.path.join(ckpt_dir, "sample_output.json")
        json.dump(payload, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"Saved sample inference to: {out_path}")
        return out_path
    except Exception as e:
        print(f"Sample inference failed: {e}")
        return None

def get_unwrapped_base_model(model, accelerator, config):
    unwrapped_model = accelerator.unwrap_model(model)
    base_model = unwrapped_model
    if config.lora_rank > 0:
        if hasattr(unwrapped_model, 'base_model'):
            base_model = unwrapped_model.base_model
            if hasattr(base_model, 'model'):
                base_model = base_model.model
    return base_model

def kl_divergence_loss(teacher_logits, student_logits, temperature=1.0):
    """
    计算 Teacher 和 Student logits 之间的 KL 散度损失（软标签蒸馏）。
    
    Args:
        teacher_logits: Teacher 模型的 logits, shape (Batch, Length, Vocab)
        student_logits: Student 模型的 logits, shape (Batch, Length, Vocab)
        temperature: 温度参数，用于软化概率分布
    
    Returns:
        KL 散度损失值
    """
    # 对 logits 应用温度缩放
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    
    # 计算 KL 散度: KL(teacher || student)
    # F.kl_div 期望 input 是 log-probabilities，target 是 probabilities
    kl_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')
    
    # 乘以 temperature^2 来保持梯度尺度（Hinton et al., 2015）
    return kl_loss * (temperature ** 2)


def similarity_preserving_loss(teacher_features, student_features, eps=1e-8):
    """
    实现 Similarity-Preserving (SP) Knowledge Distillation Loss。
    参考论文: arXiv:1907.09682
    
    计算步骤:
    1. Reshape: 将特征展平为 (Batch, Dim)
    2. Similarity: 计算 Q * Q^T 得到 (Batch, Batch) 的相似性矩阵
    3. Normalize: 对相似性矩阵进行行方向的 L2 归一化
    4. Loss: 计算教师和学生相似性矩阵的 Frobenius 范数距离
    """
    # 1. 获取 Batch 大小
    b = teacher_features.size(0)
    
    # 2. Reshape to (Batch, Vector_Dim)
    # 无论输入是 (B, L, D) 还是其他维度，都展平为每个样本一个向量
    teacher_view = teacher_features.view(b, -1)
    student_view = student_features.view(b, -1)
    
    # 3. Compute Pairwise Similarity Matrix G = Q * Q^T (Result: Batch x Batch)
    # 这一步捕捉了 Batch 内部样本之间的关系
    teacher_g = torch.mm(teacher_view, teacher_view.t())
    student_g = torch.mm(student_view, student_view.t())
    
    # 4. Row-wise L2 Normalization
    # 论文公式 (2) 和 (3): G[i,:] = G_tilde[i,:] / ||G_tilde[i,:]||_2
    teacher_g_norm = teacher_g / (teacher_g.norm(p=2, dim=1, keepdim=True) + eps)
    student_g_norm = student_g / (student_g.norm(p=2, dim=1, keepdim=True) + eps)
    
    # 5. Compute Loss: Mean Squared Frobenius Norm difference
    # 论文公式 (4)
    diff = teacher_g_norm - student_g_norm
    loss = torch.norm(diff, p='fro') ** 2 / (b ** 2)
    return loss

def compute_loss(batch, base_model, config):
    input_ids = batch['input_ids']
    batchsize, prompt_length = input_ids.shape
    max_num_blocks = config.max_gen_length // config.block_length
    total_loss = None
    total_sp_loss = 0.0
    total_kl_loss = 0.0
    num_loss_steps = 0
    
    # 获取配置参数
    kl_temperature = getattr(config, 'kl_temperature', 1.0)
    kl_weight = getattr(config, 'kl_weight', 1.0)
    sp_weight = getattr(config, 'sp_weight', 1.0)
    
    with torch.no_grad():
        prefill_outputs = base_model.prefill_phase(input_ids, config.block_length)
        past_key_values = prefill_outputs['past_key_values']
        prefill_logits = prefill_outputs['logits'].detach()
        
    block_template = torch.full((batchsize, config.block_length), base_model.mask_id, 
                                dtype=torch.long, device=input_ids.device)
                                
    # Perform an initial forward pass on the masked block to obtain block-length logits
    with torch.no_grad():
        init_outputs = base_model(
            input_ids=block_template,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            output_hidden_states=True
        )
        init_logits = init_outputs.logits.detach()
        
    # Initialize teacher and student blocks
    teacher_block = base_model.unmask_function_greedy(
        logits=init_logits,
        x=block_template.clone(),
        threshold=config.threshold_high
    )
    student_block = base_model.unmask_function_greedy(
        logits=init_logits,
        x=block_template.clone(),
        threshold=config.threshold_low
    )
    
    # Cleanup temporary tensors
    del init_outputs, init_logits, prefill_logits, prefill_outputs
    
    for i in range(max_num_blocks):
        # --- Teacher Forward ---
        with torch.no_grad():
            teacher_outputs = None
            while (teacher_block == base_model.mask_id).any():
                teacher_outputs = base_model(input_ids=teacher_block, past_key_values=past_key_values, use_cache=True, return_dict=True, output_hidden_states=True)
                teacher_block = base_model.unmask_function_greedy(logits=teacher_outputs.logits, x=teacher_block, threshold=config.threshold_high)
            
            if teacher_outputs is None:
                teacher_outputs = base_model(input_ids=teacher_block, past_key_values=past_key_values, use_cache=True, return_dict=True, output_hidden_states=True)
            
            teacher_hidden_states = teacher_outputs.hidden_states[-1].detach()  # Last layer hidden states
            teacher_logits = teacher_outputs.logits.detach()  # Teacher logits for KL loss
            teacher_kv = teacher_outputs.past_key_values
        
        # --- Student Forward ---
        student_outputs = None
        while (student_block == base_model.mask_id).any():
            student_outputs = base_model(
                input_ids=student_block,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True
            )
            student_block = base_model.unmask_function_greedy(
                logits=student_outputs.logits, 
                x=student_block, 
                threshold=config.threshold_low
            )
        
        if student_outputs is None:
            student_outputs = base_model(
                input_ids=student_block,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True
            )
        
        # --- Similarity-Preserving Distillation ---
        student_hidden_states = student_outputs.hidden_states[-1]
        student_logits = student_outputs.logits
        
        # 获取有效 Token 的掩码
        mask_bool = (teacher_block != base_model.mask_id)
        
        if mask_bool.any():
            # 1. 预处理：Mask 掉无效的位置（置零），以免干扰相似性计算
            # 保持 (Batch, Length, Dim) 的形状，因为 SP 需要计算 Batch 间的相似性
            mask_expanded = mask_bool.unsqueeze(-1).expand_as(student_hidden_states)
            
            teacher_acts = teacher_hidden_states * mask_expanded.float()
            student_acts = student_hidden_states * mask_expanded.float()
            
            # 2. 计算 SP Loss
            # 注意：SP 依赖于 Batch > 1 来计算样本间的关系。
            # 如果 batchsize=1，SP loss 将始终为 0 (归一化后都是 1)。
            if batchsize > 1:
                sp_loss = similarity_preserving_loss(teacher_acts, student_acts)
            else:
                # Fallback for batchsize=1 (Standard MSE or skip)
                # SP Loss 在 Batch=1 时无定义（无法计算样本间关系），这里退化为简单的特征对齐
                sp_loss = F.mse_loss(student_acts, teacher_acts, reduction='mean')
            
            # 3. 计算 KL Divergence Loss (Logits 交叉熵蒸馏)
            # 只在有效 token 位置计算 KL 损失
            # 创建 logits 的 mask (Batch, Length, Vocab)
            mask_logits = mask_bool.unsqueeze(-1).expand_as(student_logits)
            
            # 将无效位置的 logits 设置为很小的值，使其 softmax 后贡献最小
            masked_teacher_logits = teacher_logits.masked_fill(~mask_logits, -1e9)
            masked_student_logits = student_logits.masked_fill(~mask_logits, -1e9)
            
            # 只对有效 token 位置计算 KL 损失
            # Reshape to (Batch * Length, Vocab) for kl_div
            valid_positions = mask_bool.sum().item()
            if valid_positions > 0:
                # 展平并过滤有效位置
                flat_mask = mask_bool.view(-1)  # (Batch * Length,)
                flat_teacher = teacher_logits.view(-1, teacher_logits.size(-1))[flat_mask]  # (valid_positions, Vocab)
                flat_student = student_logits.view(-1, student_logits.size(-1))[flat_mask]  # (valid_positions, Vocab)
                
                kl_loss = kl_divergence_loss(flat_teacher, flat_student, temperature=kl_temperature)
            else:
                kl_loss = torch.tensor(0.0, device=student_logits.device)
            
            # 4. 组合损失
            step_loss = sp_weight * sp_loss + kl_weight * kl_loss
            total_loss = step_loss if total_loss is None else total_loss + step_loss
            total_sp_loss += sp_loss.item()
            total_kl_loss += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
        
        # 清理
        del student_hidden_states, teacher_hidden_states, student_outputs, student_logits, teacher_logits
        num_loss_steps += 1
        
        if (teacher_block == config.train_eos_token_id).any():
            break
            
        past_key_values = teacher_kv
        teacher_block = block_template.clone()
        student_block = block_template.clone()
    
    # --- Loss Aggregation ---
    if num_loss_steps > 0 and total_loss is not None:
        avg_loss = total_loss / num_loss_steps
        loss_tensor = avg_loss
        # 记录损失组件
        loss_components = {
            "sp_loss": total_sp_loss / num_loss_steps,
            "kl_loss": total_kl_loss / num_loss_steps,
        }
        student_loss_val = loss_tensor.item()
    else:
        loss_tensor = torch.tensor(0.0, device=base_model.device, requires_grad=True)
        student_loss_val = 0.0
        loss_components = {"sp_loss": 0.0, "kl_loss": 0.0}
    
    return {
        "loss_tensor": loss_tensor,
        "student_loss": student_loss_val,
        "num_steps": num_loss_steps,
        "loss_components": loss_components,
    }

def training_step_accum(batch, model, optimizer, lr_scheduler, accelerator, config, micro_step):
    model.train()
    base_model = get_unwrapped_base_model(model, accelerator, config)
    loss_dict = compute_loss(batch, base_model, config)
    accelerator.backward(loss_dict["loss_tensor"])
    
    grad_norm_val = 0.0
    if (micro_step % max(1, config.gradient_accumulation_steps)) == 0:
        grad_norm_ret = accelerator.clip_grad_norm_(model.parameters(), config.clip_grad)
        grad_norm_val = (grad_norm_ret.item() if isinstance(grad_norm_ret, torch.Tensor) 
                         else float(grad_norm_ret) if grad_norm_ret else 0.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    
    metrics = {
        "loss": loss_dict["loss_tensor"].item(),
        "student_loss": loss_dict["student_loss"],
        "lr": lr_scheduler.get_last_lr()[0],
        "grad_norm": grad_norm_val,
    }
    # 添加损失组件
    metrics.update(loss_dict.get("loss_components", {}))
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Feature Distillation Training Script")
    parser.add_argument(
        "--config_file", 
        type=str, 
        default="train_config.yaml", 
        help="Path to the training configuration YAML file."
    )
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = types.SimpleNamespace(**config_dict)
    try:
        config.lr = float(config.lr)
        config.weight_decay = float(config.weight_decay)
        config.gradient_accumulation_steps = max(1, int(getattr(config, 'gradient_accumulation_steps', 1)))
    except ValueError as e:
        print(f"Error converting config values: {e}\nPlease check lr and weight_decay in your train_config.yaml")
        raise

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with='wandb' if getattr(config, 'use_wandb', False) else None
    )

    tokenizer = setup_tokenizer(config)
    model = setup_models(config, accelerator)
    train_dataloader = setup_dataloaders(config, tokenizer, accelerator)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=tuple(config.betas), weight_decay=config.weight_decay)
    
    total_steps = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps) * config.total_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * config.warmup_ratio), num_training_steps=total_steps)

    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader
    )

    # 初始化W&B追踪器
    if getattr(config, 'use_wandb', False):
        run_name = getattr(config, 'wandb_run_name', None)
        accelerator.init_trackers(
            project_name=getattr(config, 'wandb_project', 'step-distillation'),
            config=vars(config),
            init_kwargs={"wandb": {"name": run_name}} if run_name else None
        )
    
    # 创建本次训练的根目录
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(config.output_dir, f"distillation_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    if accelerator.is_main_process:
        print(f"\n--- Starting Training ---")
        print(f"Run directory: {run_dir}")
    
    global_step = 0

    for epoch in range(config.total_epochs):
        model.train()
        train_pbar = tqdm(train_dataloader, disable=not accelerator.is_main_process,
                         desc=f"Epoch {epoch+1}/{config.total_epochs} [Train]", leave=True)
        train_epoch_total_loss = train_epoch_total_steps = 0
        
        for batch in train_pbar:
            global_step += 1
            metrics = training_step_accum(batch, model, optimizer, lr_scheduler, 
                                         accelerator, config, micro_step=global_step)
            train_epoch_total_loss += metrics['loss']
            train_epoch_total_steps += 1
            
            # W&B 日志
            if getattr(config, 'use_wandb', False):
                log_dict = {
                    "train/loss": metrics['loss'], 
                    "train/total_loss": metrics['student_loss'],
                    "train/lr": metrics['lr'], 
                    "train/grad_norm": metrics['grad_norm'], 
                    "epoch": epoch + 1
                }
                # 添加详细的损失组件
                if 'sp_loss' in metrics:
                    log_dict["train/sp_loss"] = metrics['sp_loss']
                if 'kl_loss' in metrics:
                    log_dict["train/kl_loss"] = metrics['kl_loss']
                accelerator.log(log_dict, step=global_step)
            
            # 保存 checkpoint
            if (global_step % config.save_steps == 0) and (global_step % config.gradient_accumulation_steps == 0):
                if accelerator.is_main_process:
                    print(f"\nStep {global_step}: Saving model checkpoint...")
                ckpt_dir = save_checkpoint(accelerator, config.output_dir, global_step, epoch, 
                                          prefix="step", tokenizer=tokenizer, config=config, run_dir=run_dir)
                if getattr(config, 'use_wandb', False):
                    accelerator.log({"checkpoint/dir": ckpt_dir}, step=global_step)
                evaluate_and_save_sample(accelerator, tokenizer, model, ckpt_dir, config, global_step, epoch)
            
            # 打印训练日志
            if accelerator.is_main_process:
                log_msg = f"[Train] step {global_step} | loss={metrics['loss']:.6f} "
                if 'sp_loss' in metrics:
                    log_msg += f"sp={metrics['sp_loss']:.6f} "
                if 'kl_loss' in metrics:
                    log_msg += f"kl={metrics['kl_loss']:.6f} "
                log_msg += f"lr={metrics['lr']:.2e}"
                tqdm.write(log_msg)
        
        # 计算训练平均损失
        train_tensors = [torch.tensor(v, device=accelerator.device, dtype=torch.float32 if i == 0 else torch.int64)
                        for i, v in enumerate([train_epoch_total_loss, train_epoch_total_steps])]
        global_train_total, global_train_steps = [accelerator.gather(t).sum().item() for t in train_tensors]
        
        if global_train_steps > 0:
            avg_train_loss = global_train_total / global_train_steps
            if accelerator.is_main_process:
                print(f"Epoch {epoch+1} Train Avg Loss: {avg_train_loss:.4f}")
            if getattr(config, 'use_wandb', False):
                accelerator.log({"train/avg_loss_epoch": avg_train_loss, "epoch": epoch + 1}, step=global_step)

    if accelerator.is_main_process:
        print("\n--- Training Complete ---\nSaving final model...")
    
    final_epoch = epoch if 'epoch' in locals() else 0
    final_dir = save_checkpoint(accelerator, config.output_dir, global_step, final_epoch, 
                               prefix="final", tokenizer=tokenizer, config=config, run_dir=run_dir)
    
    if getattr(config, 'use_wandb', False):
        accelerator.log({"final_checkpoint/dir": final_dir}, step=global_step)
    
    evaluate_and_save_sample(accelerator, tokenizer, model, final_dir, config, global_step, final_epoch)

if __name__ == "__main__":
    main()