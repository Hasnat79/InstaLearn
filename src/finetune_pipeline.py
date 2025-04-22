from teacher_model import TeacherModel
from student_model import StudentModel
from pubmed_cardio_loader import PubMedCardioLoader
from utils import prepare_finetuning_dataset
import huggingface_hub

from datasets import Dataset
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning pipeline for student-teacher models")
    parser.add_argument("--abstract_count", type=int, default=None, help="Number of abstracts to load from the dataset")
    parser.add_argument("--train_student", action="store_true", help="Whether to train the student model")
    parser.add_argument("--finetune_dataset_path", type=str, default="finetuning_dataset", help="Path to save/load finetuning dataset")
    parser.add_argument("--evaluate_student", action="store_true", help="Whether to evaluate the student model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for training the student model")
    return parser.parse_args()

def main():
    args = parse_args()
    teacher_model = TeacherModel("meta-llama/Llama-3.2-1B-Instruct")
    student_model = StudentModel("meta-llama/Llama-3.2-1B-Instruct")


    pubmed_cardio_dataset = PubMedCardioLoader("hasnat79/pubmed25_cardio")
    print(f"size of dataset: {len(pubmed_cardio_dataset)}")
    # loading 5 abstracts from the dataset
    if args.abstract_count:
        n = args.abstract_count
    else: 
        n = len(pubmed_cardio_dataset)

    test_corpus = []
    for item in tqdm(pubmed_cardio_dataset, desc="Loading abstracts", total= len(pubmed_cardio_dataset)):
        test_corpus.append(item["Abstract"])
        if len(test_corpus) >= n:
            break
    print(f"Loaded {len(test_corpus)} abstracts from the dataset")
    

    # Evaluate the student model on the test_corpus with the help of teacher model
    if args.evaluate_student:
        print(f"Evaluating the student model on the test_corpus with the help of teacher model")
        chunks_for_finetuning, evaluation_details = student_model.evaluate(teacher_model, test_corpus)
        # save evaluation results of 
        with open(f"evaluation_results_{len(test_corpus)}_abstracts.json", "w") as f:
            import json
            json.dump(evaluation_details, f, indent=4)
        print(f"\nEvaluation Completed! {len(chunks_for_finetuning)} of {len(test_corpus)} abstracts selected for finetuning")
        if chunks_for_finetuning:
            print(f"Preparing the dataset for finetuning")
            finetuning_dataset = prepare_finetuning_dataset(chunks_for_finetuning)
            print(f"Finetuning dataset prepared with {len(finetuning_dataset)} examples.")
        # save finetuning dataset
            # finetuning_dataset.save_to_disk(f"finetuning_dataset_{len(test_corpus)}_abstracts")
            finetuning_dataset.save_to_disk(f"{args.finetune_dataset_path}")
            print(f"Finetuning dataset saved to disk.")

    if args.train_student:
        finetuning_dataset = Dataset.load_from_disk(args.finetune_dataset_path)
        print(finetuning_dataset)
        # trains and saves the student model
        trained_student_model = student_model.train_lora(finetuning_dataset, num_epochs=args.epochs)
        






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