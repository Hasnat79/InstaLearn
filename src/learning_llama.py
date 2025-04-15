
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from datasets import Dataset
import json
from peft import get_peft_model, LoraConfig, TaskType

import huggingface_hub
huggingface_hub.login("HF-TOKEN")

class EfficientFinetuningPipeline:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the efficient finetuning pipeline with a teacher and student model.

        Args:
            model_name: The name of the model to use (same for teacher and student in this POC)
            device: The device to run the model on
        """
        self.device = device

        print(f"Loading model on {device}...")
        # Load the tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set up padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Make sure the tokenizer has a mask token - add if it doesn't
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({'mask_token': '[MASK]'})

        # Then load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        ).to(device)

        # Resize embeddings if needed to account for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        print("Model loaded successfully.")

        # Configure thresholds
        self.accuracy_threshold = 0.8  # 80% accuracy threshold for skipping finetuning

    def extract_keywords(self, text_chunk):
        """
        Teacher model role: Extract important keywords from a text chunk.

        Args:
            text_chunk: The text chunk to extract keywords from

        Returns:
            List of tuples (keyword, position)
        """
        prompt = f"""<|system|>
You are a helpful assistant that identifies the most important keywords in text passages.
For a given text, identify up to 5 of the most important keywords and their positions (0-indexed).
Return only the keywords and positions in the format: (keyword, position)
<|user|>
Text: {text_chunk}
<|assistant|>
"""
        # print(f"prompt:\n{prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=100,
                temperature=0.0,
                do_sample=False
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(f"response:\n{response}")

        # Extract the assistant's response
        assistant_response = response.split("<|assistant|>")[-1].strip()
        # print(f"assistant_response:\n{assistant_response}")

        # Parse the keywords and positions using regex
        keywords = []
        pattern = r'\(([^,]+),\s*(\d+)\)'
        matches = re.findall(pattern, assistant_response)
        # print(f"matches:\n{matches}")
        for match in matches:
            keyword, position = match[0], int(match[1])
            keywords.append((keyword, position))

        return keywords

    def predict_masked_word(self, sentence, masked_word_position):
        """
        Student model role: Predict the masked word in a sentence.

        Args:
            sentence: The sentence with a masked word
            masked_word_position: The position of the masked word

        Returns:
            Predicted word
        """
        prompt = f"""<|system|>
You are a helpful assistant that predicts masked words in sentences.
<|user|>
{sentence} Predict the masked word.
<|assistant|>
"""
        # print(f"prompt:\n{prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(f"response:\n{response}")

        # Extract the assistant's response (should be just the predicted word)
        assistant_response = response.split("<|assistant|>")[-1].strip()
        # print(f"assistant_response:\n{assistant_response}")

        # Take the first word of the response as the prediction
        match = re.search(r'"([^"]*)"', assistant_response)
        predicted_word = match.group(1) if match else ""
        # print(f"\npredicted_word:\n{predicted_word}")

        return predicted_word

    def create_masked_sentence(self, text_chunk, keyword, position):
        """
        Create a sentence with a masked word.

        Args:
            text_chunk: The original text chunk
            keyword: The keyword to mask
            position: The position of the keyword

        Returns:
            Sentence with the keyword masked
        """
        words = text_chunk.split()
        # print(f"words:\n{words}")
        # print(f"keyword:\n{keyword}")


        # Mask the matching keyword in the text chunk
        masked_sentence = re.sub(r'\b' + re.escape(keyword) + r'\b', '[MASK]', text_chunk, flags=re.IGNORECASE)
        return masked_sentence


    def evaluate_chunk(self, text_chunk):
        """
        Evaluate if a chunk needs finetuning based on the student model's performance.

        Args:
            text_chunk: The text chunk to evaluate

        Returns:
            needs_finetuning (bool): Whether the chunk needs finetuning
            accuracy (float): The accuracy of the student model on this chunk
            evaluation_results (list): Detailed results of each keyword prediction
        """
        keywords = self.extract_keywords(text_chunk)
        print(f"Keywords extracted: {keywords}")


        if not keywords:
            return False, 1.0, []  # No keywords found, no need for finetuning

        correct_predictions = 0
        evaluation_results = []

        for keyword, position in keywords:
            print(f"\nkeyword: {keyword}, position: {position}")
            masked_sentence = self.create_masked_sentence(text_chunk, keyword, position)
            print(f"\nmasked_sentence:\n{masked_sentence}\n")
            predicted_word = self.predict_masked_word(masked_sentence, position)
            # Check if prediction is correct
            is_correct = predicted_word.lower() == keyword.lower()

            print(f"\nkeyword: {keyword}, predicted_word: {predicted_word}, is_correct: {is_correct}")

            if is_correct:
                correct_predictions += 1

            evaluation_results.append({
                "keyword": keyword,
                "masked_sentence": masked_sentence,
                "prediction": predicted_word,
                "is_correct": is_correct
            })
        print(evaluation_results)

        # Calculate accuracy
        accuracy = correct_predictions / len(keywords) if keywords else 1.0

        # Determine if chunk needs finetuning
        needs_finetuning = accuracy < self.accuracy_threshold

        return needs_finetuning, accuracy, evaluation_results

    def prepare_finetuning_dataset(self, evaluation_details):
        """
        Prepare a dataset for finetuning from the evaluation results,
        focusing on incorrectly predicted masked words.

        Args:
            evaluation_details: List of dictionaries containing evaluation results for each chunk

        Returns:
            finetuning_dataset: Dataset object ready for finetuning
        """
        # Format the examples for masked word prediction
        formatted_examples = []

        for chunk_detail in evaluation_details:
            print(chunk_detail)
            if not chunk_detail["needs_finetuning"]:
                continue  # Skip chunks that don't need finetuning

            # Extract incorrectly predicted keywords
            for result in chunk_detail["evaluation_results"]:
                if not result["is_correct"]:
                    example = {
                        "instruction": "Predict the masked word in the sentence",
                        "input": result["masked_sentence"],
                        "output": f"The masked word is {result['keyword']}"
                    }
                    formatted_examples.append(example)

        # Convert to HF Dataset
        dataset_dict = {
            "instruction": [ex["instruction"] for ex in formatted_examples],
            "input": [ex["input"] for ex in formatted_examples],
            "output": [ex["output"] for ex in formatted_examples]
        }

        finetuning_dataset = Dataset.from_dict(dataset_dict)
        return finetuning_dataset

    def format_for_finetuning(self, example):
        """
        Format examples for finetuning using Llama chat template.
        Returns a dictionary with the formatted text in a new field.
        """
        formatted_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
                {"role": "assistant", "content": example["output"]}
            ],
            tokenize=False,
            add_generation_prompt=False
        )

        # Return a dictionary with the formatted text
        return {"formatted_text": formatted_text}

    def setup_lora_finetuning(self, finetuning_dataset):
        """
        Set up LoRA finetuning for the model.

        Args:
            finetuning_dataset: Dataset to finetune on

        Returns:
            peft_model: LoRA-configured model ready for finetuning
            formatted_dataset: The dataset with chat-template formatting
        """
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        # Create PEFT model
        peft_model = get_peft_model(self.model, peft_config)

        # Apply formatting to dataset for instruction tuning
        formatted_dataset = finetuning_dataset.map(self.format_for_finetuning)

        # Create a function to tokenize the formatted text
        def tokenize_function(examples):
            return self.tokenizer(
                examples["formatted_text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )

        # Tokenize the dataset
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_dataset.column_names  # Remove the original columns
        )

        return peft_model, tokenized_dataset

def main():
    # Initialize the pipeline
    pipeline = EfficientFinetuningPipeline()

    # Example text corpus (simplified for demonstration)
    text_corpus = [
        "The capital of India is Delhi. It is a major cultural and economic center.",
        "Python is a popular programming language known for its simplicity and readability.",
        "The Great Wall of China is over 13,000 miles long. It was built to protect against invasions.",
        "Albert Einstein developed the theory of relativity which revolutionized physics.",
        "The Pacific Ocean is the largest and deepest ocean on Earth.",
    ]

    # Process each chunk
    chunks_for_finetuning = []
    evaluation_details = []

    print("Evaluating text chunks...")
    for i, chunk in enumerate(tqdm(text_corpus)):
        print(f"\n\nEvaluating chunk {i + 1}/{len(text_corpus)}: {chunk}\n")
        needs_finetuning, accuracy, results = pipeline.evaluate_chunk(chunk)

        exit()
        chunk_details = {
            "chunk_id": i,
            "text": chunk,
            "needs_finetuning": needs_finetuning,
            "accuracy": accuracy,
            "evaluation_results": results
        }

        evaluation_details.append(chunk_details)

        if needs_finetuning:
            chunks_for_finetuning.append(chunk_details)

    # Save evaluation details
    with open("evaluation_results.json", "w") as f:
        json.dump(evaluation_details, f, indent=2)

    print(f"\nEvaluation complete. {len(chunks_for_finetuning)} of {len(text_corpus)} chunks need finetuning.")
    # Prepare finetuning dataset if needed
    if chunks_for_finetuning:
        print("Preparing finetuning dataset...")
        finetuning_dataset = pipeline.prepare_finetuning_dataset(chunks_for_finetuning)

        print("Setting up LoRA finetuning...")
        peft_model, formatted_dataset = pipeline.setup_lora_finetuning(finetuning_dataset)

        print("Starting LoRA finetuning...")
        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./efficient-finetuned-model",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=True if torch.cuda.is_available() else False,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
            logging_dir="./logs",  # Directory for logs
            logging_first_step=True,  # Log the first step
            logging_strategy="steps",  # Log based on steps, not epochs
            evaluation_strategy="no",  # No evaluation during training
        )

        # Create a data collator for causal language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=pipeline.tokenizer,
            mlm=False
        )

        # Initialize Trainer
        trainer = Trainer(
            model=peft_model,
            train_dataset=formatted_dataset,
            args=training_args,
            data_collator=data_collator
        )

        # Run training
        trainer.train()
        print("Finetuning completed successfully!")

    else:
        print("No chunks need finetuning. The model already knows this content well!")

if __name__ == "__main__":
    main()

