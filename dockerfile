FROM python:3.11


WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*  # ← this line

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('cmudict')"
RUN python -c "from bert_score import score; score(['test'], ['test'], model_type='distilbert-base-uncased', lang='en')"

COPY src/  /app/src/
COPY adaptiread_model_final/ /app/adaptiread_model_final/
COPY data/ /app/data/

ENV LANG=C.UTF-8
ENV PYTHONIOENCODING=utf-8

ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/adaptiread_model_final
ENV SUBTLEX_FILE_PATH=/app/data/SUBTLEX-UK.csv

ENV TRANSFORMERS_VERBOSITY=error

ENTRYPOINT ["python", "/app/src/test_pipeline.py"]
