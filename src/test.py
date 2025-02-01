import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import numpy as np

class LlamaInformationDetector:
    def __init__(self, model_path="meta-llama/Llama-3.2-1B-Instruct"):
        """
        Initialize with Llama model
        Note: You need appropriate access to download Llama models
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize Llama model and tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", cached_dir = "cache")
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=True,  # Quantization for memory efficiency
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir  = "cache"
        )
        
        # Prepare model for LORA fine-tuning
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LORA specifically for Llama
        self.lora_config = LoraConfig(
            r=8,  # Rank dimension
            lora_alpha=32,
            target_modules=[
                "q_proj",  # Query projection
                "k_proj",  # Key projection
                "v_proj",  # Value projection
                "o_proj"   # Output projection
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Initialize training parameters
        self.learning_rate = 3e-4
        self.num_epochs = 3
        
    def identify_key_info(self, sentence):
        """
        Use Llama to identify key information in the sentence
        Returns: List of (word, index) tuples
        """
        # Prompt engineering for Llama
        prompt = f"""Identify the key factual information in this sentence by listing important words and their positions:
        Sentence: {sentence}
        Important words and their positions (starting from 0):"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
        
        # Process the response (simplified for demonstration)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse response to get key information (simplified implementation)
        words = sentence.split()
        key_info = []
        for idx, word in enumerate(words):
            if word[0].isupper():  # Simple heuristic for demonstration
                key_info.append((word, idx))
        
        return key_info
    
    def prepare_for_training(self, sentence, key_info):
        """
        Prepare the sentence for Llama fine-tuning
        """
        # Create masked version for training
        words = sentence.split()
        for word, idx in key_info:
            words[idx] = "<mask>"
        masked_sentence = " ".join(words)
        
        # Prepare training prompt
        prompt = f"""Complete the following sentence with the correct information:
        {masked_sentence}
        Original sentence:"""
        
        return prompt
    
    def fine_tune(self, training_data):
        """
        Fine-tune Llama with LORA
        training_data: List of (sentence, key_info) tuples
        """
        # Apply LORA adapters
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Prepare optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for sentence, key_info in training_data:
                # Prepare input
                prompt = self.prepare_for_training(sentence, key_info)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(training_data)}")
    
    def evaluate_knowledge(self, sentence):
        """
        Evaluate if the model has the correct knowledge
        """
        key_info = self.identify_key_info(sentence)
        prompt = self.prepare_for_training(sentence, key_info)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated, key_info

# Example usage
def main():
    # Initialize detector
    detector = LlamaInformationDetector()
    
    # Example training data
    training_data = [
        (
            "SpaceX launched Starship in April 2023",
            [("SpaceX", 0), ("Starship", 2), ("April", 4), ("2023", 5)]
        ),
        (
            "The FIFA World Cup 2022 was held in Qatar",
            [("FIFA", 1), ("World Cup", 2), ("2022", 3), ("Qatar", 6)]
        )
    ]
    
    # Fine-tune the model
    detector.fine_tune(training_data)
    
    # Test the model
    test_sentence = "OpenAI released GPT-4 in March 2023"
    generated, key_info = detector.evaluate_knowledge(test_sentence)
    print(f"Original: {test_sentence}")
    print(f"Generated: {generated}")
    print(f"Key Information: {key_info}")

if __name__ == "__main__":
    main()
