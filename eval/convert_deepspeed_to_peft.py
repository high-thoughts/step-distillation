import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig, # 假设你用的是 Lora
    get_peft_model,
    PeftModel
)
import os

# --- 1. 修改这里的路径 ---

# 你训练时用的【基础模型】路径 (比如 "meta-llama/Llama-2-7b-hf")
BASE_MODEL_PATH = "/home/ma-user/work/step-distillation/models"

# 你的 DeepSpeed checkpoint 里的【模型权重文件】路径
CHECKPOINT_PATH = "/home/ma-user/work/step-distillation/output/20251110_233925/final_234/pytorch_model/mp_rank_00_model_states.pt"

# 你想保存【最终 LoRA 适配器】的【文件夹】路径
FINAL_OUTPUT_DIR = "/home/ma-user/work/step-distillation/output/20251110_233925/final_234/lora"

# --- 2. 你的 LoRA 配置 (已按你的要求更新) ---
PEFT_CONFIG = LoraConfig(
    r=16,                             # lora_rank
    lora_alpha=32,                    # lora_alpha
    target_modules=[
        "q_proj", 
        "v_proj", 
        "o_proj"
    ],
)

# ------------------------------------------------------------------
# --- 下面的代码通常不需要修改 ---
# ------------------------------------------------------------------

# 确保输出目录存在
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

print(f"1. 正在从 {BASE_MODEL_PATH} 加载基础模型和分词器...")
# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16, # 使用你训练时的 dtype
    device_map="cpu", # 先放到 CPU
    trust_remote_code=True
)
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

print("2. 正在应用 PEFT (LoRA) 配置...")
# 将 LoRA 配置应用到基础模型
model = get_peft_model(base_model, PEFT_CONFIG)
model.to(torch.bfloat16) # 确保类型一致

print(f"3. 正在从 {CHECKPOINT_PATH} 加载 DeepSpeed 权重...")
# 加载 DeepSpeed checkpoint
try:
    full_state = torch.load(CHECKPOINT_PATH, map_location='cpu')

    if 'module' in full_state:
        state_dict = full_state['module']
        print("  > 成功从 'module' 键中提取 state_dict。")
    elif 'model' in full_state:
        state_dict = full_state['model']
        print("  > 成功从 'model' 键中提取 state_dict。")
    else:
        state_dict = full_state
        print("  > 未找到 'module' 或 'model' 键，将使用根字典。")

    # 使用 strict=False，因为我们只关心加载 LoRA 权重
    # DeepSpeed checkpoint 包含了基础模型的权重 + LoRA 权重
    # PeftModel 会智能地把它们都加载进去
    model.load_state_dict(state_dict, strict=False)
    
    print("  > 权重加载成功！")

except Exception as e:
    print(f"加载权重时出错: {e}")
    print("请检查 CHECKPOINT_PATH 和模型配置。")
    exit()

print(f"4. 正在将完整的 LoRA 适配器保存到: {FINAL_OUTPUT_DIR}")
# 这是关键一步：
# .save_pretrained() 会智能地【只】保存 LoRA 适配器权重
# 并且【自动】生成 adapter_config.json
model.save_pretrained(FINAL_OUTPUT_DIR)

# 同时保存分词器文件
tokenizer.save_pretrained(FINAL_OUTPUT_DIR)

print("---")
print("转换完成！")
print(f"你的完整 LoRA 适配器现在位于: {FINAL_OUTPUT_DIR}")
print("它现在包含了 (图2) 中的所有文件。")