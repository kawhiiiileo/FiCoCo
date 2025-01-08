#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /root/autodl-tmp/workplace/LLaVA/llava-v1.5-7b \
    --question-file /root/autodl-tmp/workplace/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /root/autodl-tmp/workplace/LLaVA/playground/data/eval/pope/val2014 \
    --answers-file /root/autodl-tmp/workplace/LLaVA/playground/data/eval/pope/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /root/autodl-tmp/workplace/LLaVA/playground/data/eval/pope/coco \
    --question-file /root/autodl-tmp/workplace/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /root/autodl-tmp/workplace/LLaVA/playground/data/eval/pope/answers/llava-v1.5-13b.jsonl
