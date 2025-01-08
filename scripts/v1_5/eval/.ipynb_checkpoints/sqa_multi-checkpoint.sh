#!/bin/bash

# 获取GPU列表，如果没有指定CUDA_VISIBLE_DEVICES则默认使用GPU 0
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# 计算GPU的数量
CHUNKS=${#GPULIST[@]}

# 设置模型路径和数据文件路径
CKPT="llava-v1.5-13b"
SPLIT="llava_test_CQM-A"

# 对每个GPU执行模型推理任务
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_science \
        --model-path ./llava-v1.5-7b \
        --question-file ./playground/data/eval/scienceqa/$SPLIT.json \
        --image-folder ./playground/data/eval/scienceqa/images/test \
        --answers-file ./playground/data/eval/scienceqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

# 等待所有GPU任务完成
wait

# 合并各个GPU生成的分块文件
output_file=./playground/data/eval/scienceqa/answers/$CKPT/merge.jsonl

# 清空输出文件（如果已存在）
> "$output_file"

# 遍历每个分块文件并将其内容追加到输出文件中
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/scienceqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# 执行评估脚本，将答案文件转化为最终结果文件
python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file $output_file \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT}_result.json

