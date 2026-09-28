FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download es_core_news_sm

# These two are the expensive network downloads (Argos's language packages,
# and Wiktionary's EN/ES dictionary exports) - copy only what they need
# (they're standalone, no app/ imports) and run them BEFORE the full source
# copy below, so an ordinary app code change reuses this layer's cache
# instead of re-downloading everything on every deploy.
COPY scripts/install_translate_models.py scripts/build_dictionary_data.py scripts/
RUN python scripts/install_translate_models.py
RUN python scripts/build_dictionary_data.py

COPY . .

ENV WORDBYWORD_DATA_DIR=/data

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
