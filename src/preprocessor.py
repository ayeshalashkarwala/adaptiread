import pandas as pd
import spacy
from textstat import textstat
import textstat as ts_module
from transformers import AutoTokenizer
import os


MODEL_CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "t5-small")
SUBTLEX_FILE_PATH = os.environ.get("SUBTLEX_FILE_PATH", "data/SUBTLEX-UK.csv")
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Downloading...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class AdaptiReadPreprocessor:
    def __init__(self):
        self.word_freq_dict = {}
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        self.load_frequency_data()

    def load_frequency_data(self):
        if os.path.exists(SUBTLEX_FILE_PATH):
            df = pd.read_csv(SUBTLEX_FILE_PATH, sep=';',on_bad_lines='skip', low_memory=False)
            if 'Spelling' in df.columns and 'LogFreq(Zipf)' in df.columns:
                self.word_freq_dict = dict(zip(df['Spelling'].astype(str).str.lower(), pd.to_numeric(df['LogFreq(Zipf)'], errors='coerce').fillna(0.0)))
            else:
                print("Columns 'Spelling' or 'LogFreq(Zipf)' not found in SUBTLEX.")

    def get_word_rarity(self, w):
        freq = self.word_freq_dict.get(w.lower(), 0.0)
        return 8.0 - freq if freq > 0 else 8.0

    def calculate_dsci(self, text):
        doc = nlp(text)
        total_rarity_score = 0.0
        word_count = 0

        for token in doc:
            if token.is_alpha and not token.is_stop:
                word_count += 1
                rarity = self.get_word_rarity(token.text)
                syllables = ts_module.textstat.syllable_count(token.text)
                word_complexity = rarity + (syllables * 1.5)
                total_rarity_score += word_complexity
        if word_count == 0:
            return 0.0
        d_sci_score = total_rarity_score / word_count
        return d_sci_score

    def preprocess_function(self, examples):
        inputs = ["simplify: " + str(doc) for doc in examples["complex_sentence"]]
        simple_sentences = []
        for doc in examples["simple_sentence"]:
            if isinstance(doc, bytes):
                simple_sentences.append(doc.decode("utf-8", errors="ignore"))
            else:
                simple_sentences.append(str(doc))
        model_inputs = self.tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
        labels = self.tokenizer(text_target=simple_sentences, max_length=MAX_TARGET_LENGTH, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["d_sci_score"] = [self.calculate_dsci(str(doc)) for doc in examples["complex_sentence"]]
        return model_inputs