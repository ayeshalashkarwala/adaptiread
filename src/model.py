import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "/adaptiread_model_final")
class AdaptiReadModel:
    def __init__(self, model_path=None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.model = None
        self.tokenizer =None
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path) and os.path.exists(os.path.join("..", self.model_path)):
            self.model_path = os.path.join("..", self.model_path)
        if os.path.exists(self.model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            print(f"Loaded model from {self.model_path}")
        else:
            print(f"Model not found at {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    
    def simplify(self, complex_sentence, do_sample=False, top_p=0.95, repetition_penalty=1.15, temperature=0.9):
        if not complex_sentence or not complex_sentence.strip():
            return ""
        input_text = "simplify: "+str(complex_sentence)
        inputs = self.tokenizer(input_text, return_tensors="pt").to('cpu')

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=128,
                do_sample=False,         
                temperature=temperature,        
                top_p=top_p,             
                repetition_penalty=repetition_penalty, # Penalize repeating words (discourages copying)
            )
        simple_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return simple_text
    
    def batch_simplify(self, text_list):
        batch_inputs = self.tokenizer(text_list, return_tensors="pt", truncation=True, padding=True).to('cpu')
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                max_length=128,  
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.15,
            )
        
        decoded_texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
        return decoded_texts
    