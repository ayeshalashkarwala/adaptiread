# AdaptiRead: Text Simplification Tool for Dyslexia

**AdaptiRead** is an intelligent NLP pipeline designed to improve text accessibility for individuals with dyslexia. Unlike standard simplification tools, AdaptiRead uses a custom complexity metric to determine *if* simplification is necessary, ensures semantic fidelity using BERTScore, and provides Explainable AI (XAI) visualizations to show *why* text was flagged as complex.

---

## The Problem
Dyslexia affects approximately 5-10% of the population, making the decoding of high-frequency syllables and rare lexical items cognitively demanding. Traditional text simplification models often over-simplify content or lose critical nuance. AdaptiRead solves this by creating a **conditional pipeline** that only intervenes when specific linguistic thresholds are breached.

## Key Features

* **Custom Complexity Metric (DSCI):** Calculates a *Dyslexic Complexity Score* based on Zipfian word frequency (from the SUBTLEX-UK corpus) and phonological syllable counts.
* **Conditional Logic:** The simplification model (T5-Small) is only triggered if the text's DSCI score exceeds a threshold of **11.0**.
* **Semantic Guardrails:** Integrated **BERTScore** evaluation ensures the simplified output retains >85% semantic similarity to the original text.
* **Explainable AI (XAI):** Uses **LIME** (Local Interpretable Model-agnostic Explanations) to visualize which specific words contributed to the complexity score.

## Methodology & Pipeline

1.  **Preprocessing & Scoring:**
    The system tokenizes input text and calculates a score using the formula:
    $$\text{Word Complexity} = \text{Rarity (Zipf)} + (\text{Syllables} \times 1.5)$$
2.  **Simplification (Seq2Seq):**
    If the average DSCI score > 11.0, the text is passed to a **T5-Small Transformer** fine-tuned on the **WikiAuto** dataset.
3.  **Validation:**
    The output is compared against the input using **BERTScore**. If $F1 < 0.85$, the model regenerates the output to prevent meaning drift.
4.  **Explanation:**
    A LIME explainer generates a feature importance plot, highlighting difficult words.

## Tech Stack

* **Language:** Python 3.13.7
* **ML & NLP:** PyTorch, Hugging Face Transformers (T5), Spacy, NLTK
* **Evaluation & XAI:** BERTScore, LIME, TextStat
* **Data Manipulation:** Pandas, NumPy, Datasets

## Project Structure
- AdaptiRead.ipynb # Main notebook containing training, inference, and LIME logic
- README.md # Project documentation

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ayeshalashkarwala/adaptiRead.git](https://github.com/ayeshalashkarwala/adaptiRead.git)
    cd AdaptiRead
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy spacy textstat transformers datasets nltk torch bert_score lime matplotlib sentencepiece tf-keras
    ```

3.  **Download Spacy model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Results Example

**Original Input:**
> "Text difficulty was operationalized through a composite index combining mean sentence length, proportion of low-frequency academic vocabulary, and a standardized readability score."

**Analysis:**
* **DSCI Score:** 12.0 (High Complexity → Simplification Triggered)
* **LIME Result:** Identified "operationalized", "vocabulary", and "aggregation" as complexity drivers.

**AdaptiRead Output:**
> "Standardized readability score was rescaled to a common metric before aggregation."

**Metric Validation:**
* **BERTScore:** 0.89 (Meaning Preserved)

## References & Datasets

* **WikiAuto Dataset:** Used for fine-tuning the Seq2Seq T5 model. This can be accessed using [Tensorflow datasets](https://www.tensorflow.org/datasets/catalog/wiki_auto).
* **SUBTLEX-UK:** Used for Zipf frequency analysis in the complexity metric. This can be accessed [here](https://shiny.psychology.nottingham.ac.uk/lpzwjv/SUBTLEX-UK/).
