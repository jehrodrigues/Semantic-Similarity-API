# Sentence Semantic Similarity API 
Sentence Transformer - Knowledge Distillation approach

The architecture consists of two models, the Teacher and the Student. The Teacher model produces sentence embeddings from source language. Using translated sentences, the Student model needs to mimic the teacher and generate sentence embeddings in the target language. This training will produce an alignment of vector spaces (source and target) making it possible to measure the **Semantic Similarity** between them.

This models works with transfer learning and needs to be initialize with some pretrained language models as BERT, GPT 2 or 3, XLM, XLNet, RoBERTa and so on. We follow the Paper and kept SBERT (english model) initializing the teacher model and XLM-R (multilingual model with 100 languages) initializing the student model.

This multilingual knowledge distilled version supports 50+ languages and more languages can be added by model extension. 

Supported languages: ar, bg, ca, cs, da, de, el, es, et, fa, fi, fr, fr-ca, gl, gu, he, hi, hr, hu, hy, id, it, ja, ka, ko, ku, lt, lv, mk, mn, mr, ms, my, nb, nl, pl, pt-pt, pt-br, ro, ru, sk, sl, sq, sr, sv, th, tr, uk, ur, vi, zh-cn, zh-tw.

## Development Requirements
- Python3.8.5
- Pip
- Virtualenv

### M.L Model Environment

```sh
MODEL_PATH=./models/
```

## Installation

```sh
virtualenv venv -p python3
source venv/bin/activate
pip install -r requirements.txt
```

## Running Localhost

`gunicorn main:app --reload -k uvicorn.workers.UvicornWorker`

## Running Tests

`to do`
`make test`

## Access Swagger Documentation

> <http://localhost:8000/docs>

## Access Redocs Documentation

> <http://localhost:8000/redoc>

## Project structure

Files related to application are in the `api` directory.
Application parts are:

    SentenceSemanticSimilaityAPI
    ├── api                                  - web related stuff.
    │   └── application                - app related files.
    │     └── handlers           - web routes and preprocessing.
    │     └── recognizer_model   - model related files.
    │   └── domain                     - layout files.
    ├── models                               - machine learning model.
    └── main.py                              - FastAPI application creation and configuration.