#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path ./llava-v1.5-7b \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/mmbench_dev_20230712 \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712 \
    --experiment llava-v1.5-13b
