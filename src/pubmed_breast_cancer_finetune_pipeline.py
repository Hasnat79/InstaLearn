import os
from teacher_model import TeacherModel
from student_model import StudentModel
from pubmed_breast_cancer_loader import PubMedBreastCancerLoader
from utils import prepare_finetuning_dataset
import transformers
import huggingface_hub
import torch
from dataclasses import dataclass, field


from datasets import Dataset
from typing import Dict, Optional

import argparse
from tqdm import tqdm
import yaml
from pathlib import Path
import sys

@dataclass
class ModelArguments:
    model: Optional[dict] = field(default_factory=dict)


@dataclass
class DataArguments:
    data: Optional[dict] = field(default_factory=dict)


@dataclass
class TrainingArguments():
    train: Optional[dict] = field(default_factory=dict)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning pipeline for student-teacher models")
    # parser.add_argument("--abstract_count", type=int, default=None, help="Number of abstracts to load from the dataset")
    parser.add_argument("--train_student", action="store_true", help="Whether to train the student model")
    parser.add_argument("--finetune_dataset_path", type=str, default="finetuning_dataset", help="Path to save/load finetuning dataset")
    parser.add_argument("--evaluate_student", action="store_true", help="Whether to evaluate the student model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for training the student model")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for loading abstracts")
    parser.add_argument("--end_idx", type=int, default=5, help="End index for loading abstracts")
    return parser.parse_args()

def main():
    # args = parse_args()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    args_dict = yaml.safe_load(Path(sys.argv[-1]).absolute().read_text())
    model_args, data_args, training_args = parser.parse_dict(args_dict)

    # print(f"model args: {model_args.model['teacher_model_id']}")
    # print(f"train args: {training_args.train['train_student']}")
    # print(f"data args: {data_args.data['pubmed_cardio_hf_data_id']}")


    # Evaluate the student model on the test_corpus with the help of teacher model
    if not os.path.exists(training_args.train['evaluation_results_path']):
        teacher_model = TeacherModel(model_args.model['teacher_model_id'])
        student_model = StudentModel(model_args.model['student_model_id'])
        pubmed_breast_cancer_dataset = PubMedBreastCancerLoader(data_args.data['pubmed_cardio_hf_data_id'])
        print(f"size of dataset: {len(pubmed_breast_cancer_dataset)}")

        # load test corpus based on start and end index
        test_corpus = []
        for i in range(data_args.data['start_idx'], data_args.data['end_idx']):
            item = pubmed_breast_cancer_dataset.dataset['train'][i]
            test_corpus.append(item["Abstract"])

        print(f"Loaded {len(test_corpus)} abstracts from the dataset")
        print(f"Evaluating the student model on the test_corpus with the help of teacher model")
        chunks_for_finetuning, evaluation_details = student_model.evaluate(teacher_model, test_corpus)
        # save evaluation results of 
        with open(training_args.train['evaluation_results_path'], "w") as f:
            import json
            json.dump(evaluation_details, f, indent=4)
        print(f"\nEvaluation Completed! {len(chunks_for_finetuning)} of {len(test_corpus)} abstracts selected for finetuning")
        if chunks_for_finetuning:
            print(f"Preparing the dataset for finetuning")
            finetuning_dataset = prepare_finetuning_dataset(chunks_for_finetuning)
            print(f"Finetuning dataset prepared with {len(finetuning_dataset)} examples.")
        # save finetuning dataset
            # finetuning_dataset.save_to_disk(f"finetuning_dataset_{len(test_corpus)}_abstracts")
            finetuning_dataset.save_to_disk(data_args.data['finetune_dataset_path'])
            print(f"Finetuning dataset saved to disk.")
    else:
        print(f"Finetuning dataset already exists at {data_args.data['finetune_dataset_path']}. Loading from disk.")
        finetuning_dataset = Dataset.load_from_disk(data_args.data['finetune_dataset_path'])
        print(f"Loaded finetuning dataset with {len(finetuning_dataset)} examples.")
    
    # free vram of gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared GPU memory.")
        
    if training_args.train['train_student']:
        # Reinitialize the student model from scratch
        student_model = StudentModel(model_args.model['student_model_id'])
        # finetuning_dataset = Dataset.load_from_disk(data_args.data['finetune_dataset_path'])
        # print(finetuning_dataset)
        # trains and saves the student model
        trained_student_model = student_model.train_lora(finetuning_dataset, num_epochs=training_args.train['epochs'],checkpoint_dir = model_args.model['checkpoint_dir']) 
        






    # student_model.train_lora(finetuning_dataset, num_epochs=15)
    # trained_student_model = StudentModel("lora_finetuned_model_final")
    # # prompt = f"""
    # # Answer the probable word(s) of the [MASK] part. It can have one or more words.
    # # {finetuning_dataset[2]['input']} 
    # # """
    # prompt = """What is the capital of india?"""
    # response =   student_model.inference(prompt)
    # print(f"response: {response}")



if __name__ == "__main__":
    main()