# AdaptiRead: Text Simplification Tool for Dyslexia

**AdaptiRead** is an intelligent, containerized NLP pipeline designed to improve text accessibility for individuals with dyslexia. Unlike standard simplification tools, AdaptiRead uses a custom complexity metric to determine *if* simplification is necessary, ensures semantic fidelity using BERTScore, and provides Explainable AI (XAI) visualizations using LIME to show *why* text was flagged as complex.

---

## The Problem
Dyslexia affects approximately 5-10% of the population, making the decoding of high-frequency syllables and rare lexical items cognitively demanding. Traditional text simplification models often over-simplify content or lose critical nuance. AdaptiRead solves this by creating a **conditional pipeline** that only intervenes when specific linguistic thresholds are breached.

## Key Features

* **Custom Complexity Metric (DSCI):** Calculates a *Dyslexic Complexity Score* based on Zipfian word frequency (from the SUBTLEX-UK corpus) and phonological syllable counts.
* **Conditional Logic:** The simplification model (T5-Small) is only triggered if the text's DSCI score exceeds a threshold of **11.0**.
* **Semantic Guardrails:** Integrated **BERTScore** evaluation (backed by a lightweight `distilbert` model) ensures the simplified output retains >85% semantic similarity to the original text.
* **Explainable AI (XAI):** Uses **LIME** (Local Interpretable Model-agnostic Explanations) to visualize which specific words contributed to the T5 model's simplification decision.
* **Containerized Deployment:** Ready-to-deploy Docker container configured with CPU-optimized PyTorch inference, pre-baked language models, and UTF-8 locale support for instant, platform-independent executions.

## Methodology & Pipeline

1.  **Preprocessing & Scoring:**
    The system tokenizes input text and calculates a score using the formula:
    $$\text{Word Complexity} = \text{Rarity (Zipf)} + (\text{Syllables} \times 1.5)$$
2.  **Simplification (Seq2Seq):**
    If the average DSCI score > 11.0, the text is passed to a **T5-Small Transformer** fine-tuned on the **WikiAuto** dataset.
3.  **Validation:**
    The output is compared against the input using **BERTScore**. If $F1 < 0.85$, the model regenerates the output to prevent meaning drift.
4.  **Explanation:**
    A LIME explainer generates feature importance weights, highlighting difficult words.

## Tech Stack

* **Language:** Python 3.11
* **Containerization:** Docker
* **ML & NLP:** PyTorch (CPU-optimized), Hugging Face Transformers (T5), SpaCy, NLTK
* **Evaluation & XAI:** BERTScore (DistilBERT), LIME, TextStat
* **Data Manipulation:** Pandas, NumPy

## Installation & Setup

### Option 1: Standard Installation (Local Run)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ayeshalashkarwala/adaptiRead.git
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download SpaCy model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

4.  **Run the script:**
    ```bash
    python src/test_pipeline.py
    ```

---

### Option 2: Docker Containerization (Recommended)

To run the pipeline in a fully isolated, platform-independent environment without setting up Python dependencies on your host machine, you can containerize it.

#### 1. Build the Docker Image
The Dockerfile is optimized to download all SpaCy, NLTK, and BERTScore models at **build time**, ensuring the container boots instantly at run time:
```bash
docker build -t adaptiread-app -f dockerfile .
```

#### 2. Run the Container
Because the T5 model checkpoints (`adaptiread_model_final`) and the SUBTLEX dataset are large, we map them as **volume bind mounts** during execution. This keeps the Docker image lightweight.

Run the container by passing a sentence directly as a command-line argument.

---

## Results Example

**Original Input:**
> "Text difficulty was operationalized through a composite index combining mean sentence length, proportion of low-frequency academic vocabulary, and a standardized readability score."

**Analysis:**
* **DSCI Score:** 12.04 (High Complexity → Simplification Triggered)
* **LIME Explanation Output:**
  ```text
  Top Influential Words:
   - difficulty: Impact 0.075
   - combining: Impact 0.042
   - frequency: Impact 0.039
   - was: Impact 0.039
   - sentence: Impact 0.038
  ```

**AdaptiRead Output:**
> "Standardized readability score was rescaled to a common metric before aggregation."

**Metric Validation:**
* **BERTScore (DistilBERT):** 0.95 (Meaning Preserved)
---

## References & Datasets

* **WikiAuto Dataset:** Used for fine-tuning the Seq2Seq T5 model. Accessible via [Tensorflow datasets](https://www.tensorflow.org/datasets/catalog/wiki_auto).
* **SUBTLEX-UK:** Used for Zipf frequency analysis in the complexity metric. Accessible [here](https://shiny.psychology.nottingham.ac.uk/lpzwjv/SUBTLEX-UK/).
