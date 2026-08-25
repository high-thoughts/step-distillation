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
    """加载主模型 (student) 和参考模型 (reference)。"""
    
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

    if accelerator.is_main_process:
        print("Loading reference model...")
    ref_model_config = AutoConfig.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
    )
    ref_model_config.use_cache = False

    reference_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        config=ref_model_config,
        torch_dtype=torch_dtype, # <-- 应用动态 dtype
        attn_implementation="eager",
        trust_remote_code=config.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    for param in reference_model.parameters():
        param.requires_grad = False
    if accelerator.is_main_process:
        print("Reference model loaded and parameters frozen.")

    if config.lora_rank > 0:
        if accelerator.is_main_process:
            print(f"Applying LoRA with rank {config.lora_rank}...")
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
    
    return model, reference_model

def setup_dataloaders(config, tokenizer, accelerator):

    train_dataset = SFTExpandDataset(
        parquet_files=config.train_files,
        tokenizer=tokenizer,
        prompt_key=config.prompt_key,
        response_key=config.response_key,
        max_length=config.max_length,
        truncation=config.truncation,
    )
    val_dataset = SFTExpandDataset(
        parquet_files=config.val_files,
        tokenizer=tokenizer,
        prompt_key=config.prompt_key,
        response_key=config.response_key,
        max_length=config.max_length,
        truncation=config.truncation,
    )
    if accelerator.is_main_process:
        print(f"Train dataset size (total): {len(train_dataset)}")
        print(f"Val dataset size (total): {len(val_dataset)}")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        collate_fn=SFTExpandDataset.collate_fn,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        collate_fn=SFTExpandDataset.collate_fn,
    )
    return train_dataloader, val_dataloader

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

def compute_loss(batch, base_model, ref_model, config):
    input_ids = batch['input_ids']
    batchsize, prompt_length = input_ids.shape
    max_num_blocks = config.max_gen_length // config.block_length
    total_loss = None
    total_ref_loss = None
    num_loss_steps = 0
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
            return_dict=True
        )
        init_logits = init_outputs.logits.detach()
    # Initialize teacher and student blocks using logits that match block length
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
    del init_outputs, init_logits
    ref_past_key_values = None
    reference_block = None
    if ref_model is not None:
        with torch.no_grad():
            ref_prefill_outputs = ref_model.prefill_phase(input_ids, config.block_length)
            ref_past_key_values = ref_prefill_outputs['past_key_values']
            # Run an initial forward pass for the reference model to get block-length logits
            reference_block = torch.full((batchsize, config.block_length), ref_model.mask_id, 
                                         dtype=torch.long, device=input_ids.device)
            ref_init_outputs = ref_model(
                input_ids=reference_block,
                past_key_values=ref_past_key_values,
                use_cache=True,
                return_dict=True
            )
            ref_init_logits = ref_init_outputs.logits.detach()
            reference_block = ref_model.unmask_function_greedy(
                logits=ref_init_logits,
                x=reference_block,
                threshold=config.threshold_high
            )
            # Cleanup temporary tensors
            del ref_init_outputs, ref_init_logits, ref_prefill_outputs
    del prefill_logits, prefill_outputs
    for i in range(max_num_blocks):
        with torch.no_grad():
            teacher_outputs = None
            while (teacher_block == base_model.mask_id).any():
                teacher_outputs = base_model(input_ids=teacher_block, past_key_values=past_key_values, use_cache=True, return_dict=True)
                teacher_block = base_model.unmask_function_greedy(logits=teacher_outputs.logits, x=teacher_block, threshold=config.threshold_high)
            if teacher_outputs is None:
                teacher_outputs = base_model(input_ids=teacher_block, past_key_values=past_key_values, use_cache=True, return_dict=True)
            teacher_logits = teacher_outputs.logits.detach()
            teacher_kv = teacher_outputs.past_key_values
            reference_logits = None
            if ref_model is not None:
                ref_outputs = None
                while (reference_block == ref_model.mask_id).any():
                    ref_outputs = ref_model(input_ids=reference_block, past_key_values=ref_past_key_values, use_cache=True, return_dict=True)
                    reference_block = ref_model.unmask_function_greedy(logits=ref_outputs.logits, x=reference_block, threshold=config.threshold_high)
                if ref_outputs is None:
                    ref_outputs = ref_model(input_ids=reference_block, past_key_values=ref_past_key_values, use_cache=True, return_dict=True)
                reference_logits = ref_outputs.logits.detach()
                ref_past_key_values = ref_outputs.past_key_values
                del ref_outputs
        student_outputs = None
        while (student_block == base_model.mask_id).any():
            student_outputs = base_model(
                input_ids=student_block,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
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
                return_dict=True
            )
        log_student = F.log_softmax(student_outputs.logits, dim=-1)
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
        mask_bool = (teacher_block != base_model.mask_id)
        if mask_bool.any():
            valid_indices = mask_bool.view(-1).nonzero(as_tuple=True)[0]
            log_student_valid = log_student.view(-1, log_student.size(-1))[valid_indices]
            teacher_probs_valid = teacher_probs.view(-1, teacher_probs.size(-1))[valid_indices]
            kl_student = F.kl_div(log_student_valid, teacher_probs_valid, reduction='batchmean', log_target=False)
            total_loss = kl_student if total_loss is None else total_loss + kl_student
            del log_student_valid, teacher_probs_valid, valid_indices
        if ref_model is not None and reference_logits is not None:
            with torch.no_grad():
                log_teacher = F.log_softmax(teacher_logits.float(), dim=-1)
                ref_probs = F.softmax(reference_logits.float(), dim=-1)
                if mask_bool.any():
                    valid_indices = mask_bool.view(-1).nonzero(as_tuple=True)[0]
                    log_teacher_valid = log_teacher.view(-1, log_teacher.size(-1))[valid_indices]
                    ref_probs_valid = ref_probs.view(-1, ref_probs.size(-1))[valid_indices]
                    kl_ref = F.kl_div(log_teacher_valid, ref_probs_valid, reduction='batchmean', log_target=False)
                    total_ref_loss = kl_ref if total_ref_loss is None else total_ref_loss + kl_ref
                    del log_teacher_valid, ref_probs_valid, valid_indices, log_teacher, ref_probs
        del log_student, teacher_probs, teacher_logits, student_outputs
        if reference_logits is not None:
            del reference_logits
        num_loss_steps += 1
        if (teacher_block == config.train_eos_token_id).any():
            break
        past_key_values = teacher_kv
        teacher_block = block_template.clone()
        student_block = block_template.clone()
        if ref_model is not None:
            reference_block = block_template.clone()
    if num_loss_steps > 0 and total_loss is not None:
        avg_student_loss = total_loss / num_loss_steps
        if ref_model is not None and total_ref_loss is not None:
            avg_ref_loss = total_ref_loss / num_loss_steps
            loss_tensor = avg_student_loss + config.beta * avg_ref_loss
            ref_loss_val = avg_ref_loss.item()
        else:
            loss_tensor = avg_student_loss
            ref_loss_val = 0.0
        student_loss_val = avg_student_loss.item()
    else:
        loss_tensor = torch.tensor(0.0, device=base_model.device, requires_grad=True)
        student_loss_val = 0.0
        ref_loss_val = 0.0
    return {
        "loss_tensor": loss_tensor,
        "student_loss": student_loss_val,
        "ref_loss": ref_loss_val,
        "num_steps": num_loss_steps,
    }

def training_step_accum(batch, model, ref_model, optimizer, lr_scheduler, accelerator, config, micro_step):
    model.train()
    base_model = get_unwrapped_base_model(model, accelerator, config)
    loss_dict = compute_loss(batch, base_model, ref_model, config)
    accelerator.backward(loss_dict["loss_tensor"])
    
    grad_norm_val = 0.0
    if (micro_step % max(1, config.gradient_accumulation_steps)) == 0:
        grad_norm_ret = accelerator.clip_grad_norm_(model.parameters(), config.clip_grad)
        grad_norm_val = (grad_norm_ret.item() if isinstance(grad_norm_ret, torch.Tensor) 
                         else float(grad_norm_ret) if grad_norm_ret else 0.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    
    return {
        "loss": loss_dict["loss_tensor"].item(),
        "student_loss": loss_dict["student_loss"],
        "ref_loss": loss_dict["ref_loss"],
        "lr": lr_scheduler.get_last_lr()[0],
        "grad_norm": grad_norm_val,
    }

def validation_step(batch, model, ref_model, accelerator, config):
    model.eval()
    with torch.no_grad():
        loss_dict = compute_loss(batch, get_unwrapped_base_model(model, accelerator, config), ref_model, config)
    
    final_loss = loss_dict["student_loss"] + (config.beta * loss_dict["ref_loss"] if ref_model else 0)
    return {"loss": final_loss, **{k: loss_dict[k] for k in ["student_loss", "ref_loss", "num_steps"]}}

def main():
    parser = argparse.ArgumentParser(description="Block Distillation Training Script")
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
        config.beta = float(config.beta)
        config.gradient_accumulation_steps = max(1, int(getattr(config, 'gradient_accumulation_steps', 1)))
        config.val_step = max(1, int(getattr(config, 'val_step', 1)))
    except ValueError as e:
        print(f"Error converting config values: {e}\nPlease check lr, weight_decay, and beta in your train_config.yaml")
        raise

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        deepspeed_plugin=DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=config.gradient_accumulation_steps),
        log_with='wandb' if getattr(config, 'use_wandb', False) else None
    )

    tokenizer = setup_tokenizer(config)
    model, reference_model = setup_models(config, accelerator)
    train_dataloader, val_dataloader = setup_dataloaders(config, tokenizer, accelerator)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=tuple(config.betas), weight_decay=config.weight_decay)
    
    total_steps = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps) * config.total_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * config.warmup_ratio), num_training_steps=total_steps)

    model, optimizer, lr_scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, val_dataloader
    )

    reference_model = reference_model.to(accelerator.device)

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
    
    last_val_loss = last_val_student_loss = last_val_ref_loss = None
    global_step = 0

    for epoch in range(config.total_epochs):
        model.train()
        train_pbar = tqdm(train_dataloader, disable=not accelerator.is_main_process,
                         desc=f"Epoch {epoch+1}/{config.total_epochs} [Train]", leave=True)
        train_epoch_total_loss = train_epoch_total_steps = 0
        
        for batch in train_pbar:
            global_step += 1
            metrics = training_step_accum(batch, model, reference_model, optimizer, lr_scheduler, 
                                         accelerator, config, micro_step=global_step)
            train_epoch_total_loss += metrics['loss']
            train_epoch_total_steps += 1
            
            # W&B 日志
            if getattr(config, 'use_wandb', False):
                accelerator.log({
                    "train/loss": metrics['loss'], "train/kl_student": metrics['student_loss'],
                    "train/kl_ref": metrics['ref_loss'], "train/lr": metrics['lr'],
                    "train/grad_norm": metrics['grad_norm'], "epoch": epoch + 1
                }, step=global_step)
            
            # 保存 checkpoint
            if (global_step % config.save_steps == 0) and (global_step % config.gradient_accumulation_steps == 0):
                if accelerator.is_main_process:
                    print(f"\nStep {global_step}: Saving model checkpoint...")
                ckpt_dir = save_checkpoint(accelerator, config.output_dir, global_step, epoch, 
                                          prefix="step", tokenizer=tokenizer, config=config, run_dir=run_dir)
                if getattr(config, 'use_wandb', False):
                    accelerator.log({"checkpoint/dir": ckpt_dir}, step=global_step)
                evaluate_and_save_sample(accelerator, tokenizer, model, ckpt_dir, config, global_step, epoch)
            # 按步验证
            if config.val_step > 0 and (global_step % config.val_step == 0) and (global_step % config.gradient_accumulation_steps == 0):
                if accelerator.is_main_process:
                    print(f"\nStep {global_step}: 进行按步验证...")
                try:
                    model.eval()
                    total_val_loss = total_val_stu_loss = total_val_ref_loss = total_val_steps = 0
                    val_pbar = tqdm(val_dataloader, disable=not accelerator.is_main_process,
                                   desc=f"Step {global_step} [Val]", leave=False)
                    
                    with torch.no_grad():
                        for batch in val_pbar:
                            val_metrics = validation_step(batch, model, reference_model, accelerator, config)
                            total_val_loss += val_metrics['loss']
                            total_val_stu_loss += val_metrics['student_loss']
                            total_val_ref_loss += val_metrics['ref_loss']
                            total_val_steps += 1
                            
                            if accelerator.is_main_process:
                                val_pbar.set_postfix(
                                    batch_loss=f"{val_metrics['loss']:.10f}",
                                    avg=f"{total_val_loss / total_val_steps:.10f}",
                                    stu=f"{total_val_stu_loss / total_val_steps:.10f}",
                                    ref=f"{total_val_ref_loss / total_val_steps:.10f}"
                                )
                    
                    # 聚合跨进程结果
                    local_vals = [torch.tensor(v, device=accelerator.device, dtype=torch.float32 if i < 3 else torch.int64)
                                 for i, v in enumerate([total_val_loss, total_val_stu_loss, total_val_ref_loss, total_val_steps])]
                    global_vals = [accelerator.gather(v).sum().item() for v in local_vals]
                    
                    if global_vals[3] > 0:
                        last_val_loss, last_val_student_loss, last_val_ref_loss = [v / global_vals[3] for v in global_vals[:3]]
                        if accelerator.is_main_process:
                            print(f"按步验证平均损失: {last_val_loss:.4f}")
                            print(f"  Val Stu: {last_val_student_loss:.4f}, Val Ref: {last_val_ref_loss:.4f}")
                        if getattr(config, 'use_wandb', False):
                            accelerator.log({
                                "val/loss": last_val_loss, "val/stu": last_val_student_loss,
                                "val/ref": last_val_ref_loss, "epoch": epoch + 1, "global_step": global_step
                            }, step=global_step)
                    elif accelerator.is_main_process:
                        print("按步验证跳过: 跨进程无批次。")
                except Exception as e:
                    if accelerator.is_main_process:
                        print(f"按步验证失败: {e}")
            # 打印训练日志
            if accelerator.is_main_process:
                val_str = f"{last_val_loss:.10f}" if last_val_loss else "—"
                val_stu_str = f"{last_val_student_loss:.10f}" if last_val_student_loss else "—"
                val_ref_str = f"{last_val_ref_loss:.10f}" if last_val_ref_loss else "—"
                tqdm.write(f"[Train] step {global_step} | loss={metrics['loss']:.10f} "
                          f"kl_stu={metrics['student_loss']:.4f} kl_ref={metrics['ref_loss']:.10f} "
                          f"lr={metrics['lr']:.10f} | val={val_str} val_stu={val_stu_str} val_ref={val_ref_str}")
        
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