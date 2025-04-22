#!/bin/bash



cd /scratch/group/instalearn/InstaLearn/src

python finetune_pipeline.py --abstract_count 100 \
--evaluate_student \
--train_student \
--finetune_dataset_path /scratch/group/instalearn/InstaLearn/src/finetuning_dataset_100_abstracts \
--epochs 15