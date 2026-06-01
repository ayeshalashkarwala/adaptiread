import os
import sys
from preprocessor import AdaptiReadPreprocessor
from utils import semantic_analysis, explain_simplification
from model import AdaptiReadModel


def run_pipeline(complex_sentence):
    preprocessor = AdaptiReadPreprocessor()
    model = AdaptiReadModel()
    dsci_score = preprocessor.calculate_dsci(complex_sentence)
    
    print(f"Original Sentence: {complex_sentence}")
    print(f"DSCI Score: {dsci_score}")

    simplified_sentence = model.simplify(complex_sentence)
    print(f"Simplified Sentence: {simplified_sentence}")

    bert_score = semantic_analysis(simplified_sentence, complex_sentence)
    print(f"BERT Score: {bert_score}")

    explanation = explain_simplification(model.model, model.tokenizer, complex_sentence, simplified_sentence, num_samples=10)
    print(f"Explanation: {explanation}")

    print("Top Influential Words:")
    for item in explanation[:5]:
        print(f" - {item['word']}: Impact {item['impact']:.3f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sentence = sys.argv[1]
    else:
        sentence = input("Enter a sentence to simplify: ")
        
    if sentence.strip():
        run_pipeline(sentence)
    else:
        print("Error: No sentence was provided.")


