from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch
import re
import json 
from utils import create_masked_sentence


class TeacherModel:
    def __init__(self, model_name = "meta-llama/Llama-3.2-1B-Instruct"): 
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Loading teacher model on {self.device}...")
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
        # Resize embeddings if needed to account for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        print("Teacher model loaded successfully.")

    def process_keywords(self, response):
        prompt = f"""<|system|>
You are a helpful assistant identify list of keywords in a text and converts them as a list of strings. 
<|user|>
Only return list of keywords in the following format:
["keyword1", "keyword2", "keyword3"]
Text: {response}
<|assistant|>
"""     
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=70,
            temperature=0.1,
            do_sample=True
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the assistant's response
        assistant_response = response.split("<|assistant|>")[-1].strip()
        # print(f"assistant_response:\n{assistant_response}")
        # print("processed _keywords:\n", assistant_response)
        # Use regex to extract a JSON-like array from the text
        list_pattern = r'\[(.*?)\]'
        match = re.search(list_pattern, assistant_response)
        if match:
            list_str = f"[{match.group(1)}]"
            try:
                # Try to parse it as JSON
                return json.loads(list_str)
            except json.JSONDecodeError:
                # If JSON parsing fails, fallback to simple string extraction
                keywords = re.findall(r'"([^"]*)"', match.group(1))
                return keywords
        else:
            # If no list is found, try to extract quoted strings
            keywords = re.findall(r'"([^"]*)"', assistant_response)
            return keywords
         
        
        

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
For a given text, identify up to 5 of the most important keywords
<|user|>
Text: {text_chunk}
<|assistant|>
"""
        # print(f"prompt:\n{prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(f"response:\n{response}")

        # Extract the assistant's response
        assistant_response = response.split("<|assistant|>")[-1].strip()
        # print(f"assistant_response:\n{assistant_response}")

        keyword_list = self.process_keywords(assistant_response)
        keywords = [keyword for keyword in keyword_list ]
        # print(f"type of keywords: {type(keywords)}")
        # print(f"keywords:\n{keywords}")

        
        # # Parse the keywords and positions using regex
        # keywords = []
        # pattern = r'\(([^,]+),\s*(\d+)\)'
        # matches = re.findall(pattern, assistant_response)
        # # print(f"matches:\n{matches}")
        # for match in matches:
        #     keyword, position = match[0], int(match[1])
        #     keywords.append((keyword, position))

        return keywords

