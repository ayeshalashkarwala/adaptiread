import warnings
from bert_score import score as bert_score
from lime.lime_text import LimeTextExplainer
import numpy as np
import torch


def semantic_analysis(simple_text, complex_text):
    if not simple_text or not complex_text:
        return 0.0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Precision, Recall, F1_score = bert_score([simple_text], [complex_text],model_type="distilbert-base-uncased" ,lang="en", verbose=False)
    return float(F1_score.item())


def explain_simplification(model, tokenizer, complex_text: str, simple_text: str, num_samples: int = 100):
    if not complex_text or not simple_text:
        return []
    
    target_ids = tokenizer(text_target=simple_text, return_tensors="pt").input_ids.to('cpu')
    
    def predictor(texts):
        probs = []
        for text in texts:
            if not text.strip():
                probs.append(0.0)
                continue
            
            inp = tokenizer("simplify: " + text, return_tensors="pt").to('cpu')
            with torch.no_grad():
                outputs = model(input_ids=inp.input_ids, labels=target_ids)
                loss = outputs.loss.item()
                prob = np.exp(-loss)
                probs.append(prob)
                
        return np.array([[1-p, p] for p in probs])
    explainer = LimeTextExplainer(class_names=["Irrelevant", "Relevant"])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = explainer.explain_instance(
            complex_text, 
            predictor, 
            num_features=10, 
            num_samples=num_samples
        )
        
    weights = exp.as_list()
    weights.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return [{"word": word, "impact": float(weight)} for word, weight in weights]