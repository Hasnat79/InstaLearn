from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import re 
import torch
from utils import create_masked_sentence
from tqdm import tqdm

from peft import get_peft_model, LoraConfig, TaskType,prepare_model_for_kbit_training

class StudentModel:
    def __init__(self,model_name = "meta-llama/Llama-3.2-1B-Instruct" ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Loading student model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Make sure the tokenizer has a mask token - add if it doesn't
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        print("Student model loaded successfully.")
        self.accuracy_threshold = 0.8

    def predict_masked_word(self, sentence):
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
            attention_mask=inputs["attention_mask"],
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=True
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
    def evaluate_chunk(self,teacher_model, text_chunk):
        """
        Evaluate if a chunk needs finetuning based on the student model's performance.

        Args:
            text_chunk: The text chunk to evaluate

        Returns:
            needs_finetuning (bool): Whether the chunk needs finetuning
            accuracy (float): The accuracy of the student model on this chunk
            evaluation_results (list): Detailed results of each keyword prediction
        """
        keywords = teacher_model.extract_keywords(text_chunk)
        print(f"Keywords: {keywords}")
        if not keywords:
            print(f"No keywords found in the chunk.")
            return False, 0.0, []

        correct_predictions = 0
        evaluation_results = []

        for keyword in keywords:
            masked_sentence = create_masked_sentence(text_chunk, keyword)
            print(f"\nmasked_sentence: {masked_sentence}")
            predicted_word = self.predict_masked_word(masked_sentence)
            print(f"predicted_word: {predicted_word}")
            is_correct = keyword.lower() in predicted_word.lower()
            if is_correct:
                correct_predictions += 1
            evaluation_results.append({
                "keyword": keyword,
                "masked_sentence": masked_sentence,
                "prediction": predicted_word,
                "is_correct": is_correct
            })
        accuracy = correct_predictions / len(keywords)
        needs_finetuning = accuracy < self.accuracy_threshold
        return needs_finetuning, accuracy, evaluation_results
        
    def evaluate(self, teacher_model, test_corpus): 
        """
        Evaluate the student model using keywords identified by teacher model from test_corpus.
        Args:
            teacher_model:Object - The teacher model used for evaluation
            test_corpus:str - List of abstract of pubmed cardio to evaluate
        """
        # print(f"Evaluating student model using teacher model and test corpus")
        evaluation_details = []
        chunks_for_fintuning = []
        total_chunks = len(test_corpus)
        for i, chunk in enumerate(tqdm(test_corpus, desc="Evaluating chunks", unit="chunk")):
            print(f"Abstract {i}/{total_chunks}: ")
            needs_finetuning, accuracy, evaluation_results = self.evaluate_chunk(teacher_model, chunk)
            print(f"needs_finetuning: {needs_finetuning}, accuracy: {accuracy}")
            # break
            chunk_details = {
            "chunk_id": i,
            "text": chunk,
            "needs_finetuning": needs_finetuning,
            "accuracy": accuracy,
            "evaluation_results": evaluation_results
            }
            evaluation_details.append(chunk_details)
            if needs_finetuning:
                chunks_for_fintuning.append(chunk_details)
            print("*"*50)
        return chunks_for_fintuning, evaluation_details
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
    # def train_lora(self,finetuning_dataset):
    #     """
    #     finetuning_dataset: 
    #         Dataset({
    #             features: ['instruction', 'input', 'output'],
    #             num_rows: 22
    #         })
    #     """
    #     print("Setting up LoRA finetuning...")
    #     peft_model, formatted_dataset = self.setup_lora_finetuning(finetuning_dataset)
    #     peft_model = peft_model.to(self.device)
    #     print(f"Starting up LoRA finetuning...")
    #     training_args = TrainingArguments(
    #         output_dir="./efficient-finetuned-student-model",
    #         per_device_train_batch_size=4,
    #         gradient_accumulation_steps=4,
    #         num_train_epochs=3,
    #         learning_rate=2e-4,
    #         fp16=True if torch.cuda.is_available() else False,
    #         logging_steps=10,
    #         save_strategy="epoch",
    #         report_to="none",
    #         logging_dir="./logs",  # Directory for logs
    #         logging_first_step=True,  # Log the first step
    #         logging_strategy="steps",  # Log based on steps, not epochs
    #         evaluation_strategy="no",  # No evaluation during training
    #         # Explicitly disable data parallelism
    #         no_cuda=False if torch.cuda.is_available() else True,
    #         # Use a single GPU
    #         local_rank=-1,  # Force running on a single GPU
    #         ddp_backend=None,  # Disable DDP
    #         dataloader_pin_memory=False  # Try disabling pin memory
    #     )
    #     # Create a data collator for causal language modeling
    #     # data_collator = DataCollatorForLanguageModeling(
    #     #     tokenizer=self.tokenizer,
    #     #     mlm=False
    #     # )
    #     # Create a custom data collator that ensures tensors are properly shaped
    #     def custom_data_collator(features):
    #         # Ensure all tensors are at least 1-dimensional
    #         processed_features = {}
    #         for key, value in features[0].items():
    #             stacked_values = [f[key] for f in features]
    #             if isinstance(stacked_values[0], torch.Tensor) and stacked_values[0].dim() == 0:
    #                 # Convert 0-dim tensors to 1-dim
    #                 stacked_values = [v.unsqueeze(0) if v.dim() == 0 else v for v in stacked_values]
    #             processed_features[key] = torch.stack(stacked_values) if isinstance(stacked_values[0], torch.Tensor) else stacked_values
    #         return processed_features
    #     # Initialize Trainer
    #     trainer = Trainer(
    #         model=peft_model,
    #         train_dataset=formatted_dataset,
    #         args=training_args,
    #         data_collator=custom_data_collator
    #     )
    #     trainer.train()
    #     print("Finetuning completed successfully!")

    def train_lora(self, dataset, 
                    lora_rank=16, 
                    lora_alpha=32, 
                    lora_dropout=0.05,
                    learning_rate=2e-4,
                    num_epochs=3,
                    batch_size=8,
                    gradient_accumulation_steps=4):
        """
        Train the model using LoRA fine-tuning on the given dataset.
        
        Args:
            dataset: Dataset object containing 'instruction', 'input', and 'output' fields
            lora_rank: Rank of the LoRA adaptation matrices
            lora_alpha: Alpha parameter for LoRA scaling
            lora_dropout: Dropout probability for LoRA layers
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            batch_size: Training batch size
            gradient_accumulation_steps: Number of steps to accumulate gradients
            
        Returns:
            The fine-tuned model
        """
        print("Preparing for LoRA fine-tuning...")
        
        # Prepare the model for LoRA fine-tuning
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        # Prepare the model for training
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        
        # Preprocess the dataset
        def preprocess_function(examples):
            # Format the text for instruction tuning
            combined_texts = []
            for instruction, inp, output in zip(examples["instruction"], examples["input"], examples["output"]):
                text = f"<s>[INST] {instruction}\n\n{inp} [/INST] {output}</s>"
                combined_texts.append(text)
                
            # Tokenize the texts
            tokenized_inputs = self.tokenizer(
                combined_texts,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Prepare the labels (same as input_ids for causal LM)
            tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
            
            return tokenized_inputs
        
        # Apply preprocessing to the dataset
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Define the training arguments
        training_args = TrainingArguments(
            output_dir=f"./lora_finetuned_model",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            logging_steps=100,
            save_steps=500,
            save_total_limit=2, 
            fp16=torch.cuda.is_available(),
            report_to="none",
            remove_unused_columns=False,
            label_names=["labels"],  # Add label_names to fix the warning
        )
        
        # Create the trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            ),
        )
        
        # Train the model
        print("Starting LoRA training...")
        trainer.train()
        
        #print trianing losses
        trainer.log_metrics("train", trainer.state.log_history[-1])
        trainer.save_metrics("train", trainer.state.log_history[-1])
        # trainer.save_state()
        # Save the fine-tuned model
        self.model.save_pretrained(f"./lora_finetuned_model_{len(dataset)}_samples")
        self.tokenizer.save_pretrained(f"./lora_finetuned_model_{len(dataset)}_samples")
        
        print("LoRA fine-tuning completed successfully.")
        return self.model


    def inference(self, prompt, max_length=1024, temperature=0.1):
            """
            Generate a response for the given prompt using the fine-tuned model.
            
            Args:
                prompt: The input prompt for which to generate a response
                max_length: Maximum length of the generated output
                temperature: Sampling temperature for generation
                
            Returns:
                The generated response as a string
            """
            # Format the prompt properly for inference
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            # Tokenize the prompt
            input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self.device)
            
            # Generate a response
            with torch.no_grad():
                # Tokenize with attention_mask
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=100
                )
            
            # Decode and return the generated response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            print(f"\n\ngenerated_text:\n{generated_text}")
            # Extract only the response part (after [/INST])
            response = generated_text.split("[/INST]")[-1].replace("</s>", "").strip()
            
            return response
