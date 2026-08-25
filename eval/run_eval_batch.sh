#!/bin/bash

# 评估脚本：使用不同的max_gen_length、batch_size和threshold参数运行模型评估
# max_gen_length: 256, 512, 1024
# threshold: 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8
# 使用4张GPU: 0, 1, 2, 3

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval.py"

# GPU设置
export ASCEND_RT_VISIBLE_DEVICES=0
NUM_GPUS=1

# 定义参数组合
declare -a MAX_GEN_LENGTHS=(256 512 1024)
declare -a THRESHOLDS=(0.4 0.9)

# 根据max_gen_length设置batch_size
get_batch_size() {
    local max_len=$1
    case $max_len in
        256)
            echo 16
            ;;
        512)
            echo 8
            ;;
        1024)
            echo 8
            ;;
        *)
            echo 8
            ;;
    esac
}

# 其他公共参数（可根据需要修改）
BASE_MODEL_PATH="/home/ma-user/work/step-distillation/models"
BLOCK_LENGTH=64
NUM_FEW_SHOT=0
DTYPE="bfloat16"
SEED=42

# 创建输出目录
OUTPUT_DIR="${SCRIPT_DIR}/eval_results"
mkdir -p "${OUTPUT_DIR}"

# 记录日志
LOG_FILE="${OUTPUT_DIR}/eval_run_$(date +%Y%m%d_%H%M%S).log"
echo "开始批量评估..." | tee -a "${LOG_FILE}"
echo "日志文件: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "评估配置:" | tee -a "${LOG_FILE}"
echo "  Max Gen Lengths: ${MAX_GEN_LENGTHS[@]}" | tee -a "${LOG_FILE}"
echo "  Thresholds: ${THRESHOLDS[@]}" | tee -a "${LOG_FILE}"
echo "  Total evaluations: $((${#MAX_GEN_LENGTHS[@]} * ${#THRESHOLDS[@]}))" | tee -a "${LOG_FILE}"
echo "================================" | tee -a "${LOG_FILE}"

# 计数器
eval_count=0
total_evals=$((${#MAX_GEN_LENGTHS[@]} * ${#THRESHOLDS[@]}))

# 双重循环：遍历所有max_gen_length和threshold组合
for MAX_GEN_LENGTH in "${MAX_GEN_LENGTHS[@]}"; do
    BATCH_SIZE=$(get_batch_size $MAX_GEN_LENGTH)
    
    for THRESHOLD in "${THRESHOLDS[@]}"; do
        eval_count=$((eval_count + 1))
        
        echo "" | tee -a "${LOG_FILE}"
        echo "========================================" | tee -a "${LOG_FILE}"
        echo "运行评估 #${eval_count}/${total_evals}" | tee -a "${LOG_FILE}"
        echo "参数: max_gen_length=${MAX_GEN_LENGTH}, threshold=${THRESHOLD}, batch_size=${BATCH_SIZE}" | tee -a "${LOG_FILE}"
        echo "----------------------------------------" | tee -a "${LOG_FILE}"
        
        # 设置输出文件名（包含threshold信息）
        OUTPUT_FILE="${OUTPUT_DIR}/gsm8k_eval_maxlen${MAX_GEN_LENGTH}_thr${THRESHOLD}_bs${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S).jsonl"
        ACCURACY_FILE="${OUTPUT_DIR}/accuracy_maxlen${MAX_GEN_LENGTH}_thr${THRESHOLD}_bs${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S).json"
        
        # 运行评估（使用torchrun进行多GPU分布式评估）
        torchrun \
            --nproc_per_node=${NUM_GPUS} \
            --master_port=29500 \
            "${EVAL_SCRIPT}" \
            --base_model_path "${BASE_MODEL_PATH}" \
            --checkpoint_path "/home/ma-user/work/step-distillation/feature-distillation/output/distillation_20251127_000745/checkpoint-464" \
            --output_file "${OUTPUT_FILE}" \
            --accuracy_file "${ACCURACY_FILE}" \
            --batch_size ${BATCH_SIZE} \
            --max_gen_length ${MAX_GEN_LENGTH} \
            --block_length ${BLOCK_LENGTH} \
            --threshold ${THRESHOLD} \
            --num_few_shot ${NUM_FEW_SHOT} \
            --dtype "${DTYPE}" \
            --subsample 256 \
            --seed ${SEED} 2>&1 | tee -a "${LOG_FILE}"
        
        if [ $? -eq 0 ]; then
            echo "✓ 评估 #${eval_count} 完成！" | tee -a "${LOG_FILE}"
        else
            echo "✗ 评估 #${eval_count} 失败！" | tee -a "${LOG_FILE}"
            exit 1
        fi
        
        # 短暂休息，避免GPU过热
        sleep 2
    done
done

echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "所有评估完成！" | tee -a "${LOG_FILE}"
echo "总计完成: ${eval_count}/${total_evals} 个评估" | tee -a "${LOG_FILE}"
echo "结果保存在: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# 生成结果汇总
echo "" | tee -a "${LOG_FILE}"
echo "生成结果汇总..." | tee -a "${LOG_FILE}"
SUMMARY_FILE="${OUTPUT_DIR}/evaluation_summary_$(date +%Y%m%d_%H%M%S).txt"

echo "评估结果汇总" > "${SUMMARY_FILE}"
echo "评估时间: $(date)" >> "${SUMMARY_FILE}"
echo "========================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# 按max_gen_length分组汇总
for MAX_GEN_LENGTH in "${MAX_GEN_LENGTHS[@]}"; do
    echo "Max Gen Length: ${MAX_GEN_LENGTH}" >> "${SUMMARY_FILE}"
    echo "----------------------------------------" >> "${SUMMARY_FILE}"
    
    for THRESHOLD in "${THRESHOLDS[@]}"; do
        # 查找对应的accuracy文件
        accuracy_files=$(ls -t "${OUTPUT_DIR}"/accuracy_maxlen${MAX_GEN_LENGTH}_thr${THRESHOLD}_*.json 2>/dev/null | head -1)
        
        if [ -n "$accuracy_files" ]; then
            # 提取准确率与平均循环次数
            accuracy=$(python3 -c "import json; print(json.load(open('$accuracy_files'))['accuracy_percent'])" 2>/dev/null || echo "N/A")
            avg_loops=$(python3 -c "import json; print(json.load(open('$accuracy_files')).get('avg_block_decode_iterations_per_sample','N/A'))" 2>/dev/null || echo "N/A")
            echo "  Threshold ${THRESHOLD}: ${accuracy} | Avg Loops: ${avg_loops}" >> "${SUMMARY_FILE}"
        else
            echo "  Threshold ${THRESHOLD}: 未找到结果" >> "${SUMMARY_FILE}"
        fi
    done
    
    echo "" >> "${SUMMARY_FILE}"
done

echo "汇总文件已保存: ${SUMMARY_FILE}" | tee -a "${LOG_FILE}"
cat "${SUMMARY_FILE}" | tee -a "${LOG_FILE}"
