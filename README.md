---
title: Trust Review
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Trust Review — Fake Review Detection System

A machine learning system that classifies product reviews as **Real** or **Fake** using traditional ML models and a DistilBERT transformer. Built as Capstone Project 2 for B.S. Computer Science and Data Analytics at IIT Patna.

---

## Demo

🚀 **[Live Demo — huggingface.co/spaces/rudra05/trust-review](https://huggingface.co/spaces/rudra05/trust-review)**

---

## Features

- **Single Review Analysis** — paste any review, get a Real/Fake label with confidence score
- **Bulk CSV/XLSX Upload** — analyze thousands of reviews at once with downloadable labeled output
- **URL Scraping** — enter any product page URL to scrape and analyze reviews automatically
- **Multiple Models** — choose between Logistic Regression, SVM, Random Forest, or DistilBERT

---

## Tech Stack

| Layer         | Technology                                        |
| ------------- | ------------------------------------------------- |
| Backend API   | FastAPI + Uvicorn                                 |
| ML Models     | Scikit-learn (LR, SVM, Random Forest)             |
| Deep Learning | HuggingFace DistilBERT                            |
| Frontend      | Vanilla JS + HTML/CSS                             |
| Datasets      | OPUS Deceptive Opinion Spam, Amazon Reviews, Yelp |

---

## Project Structure

```
Capstone Project/
├── app/
│   ├── api/
│   │   ├── routes.py          # FastAPI endpoints
│   │   └── schemas.py         # Pydantic request/response models
│   ├── core/
│   │   ├── config.py          # Paths and settings
│   │   └── model_loader.py    # Model loading at startup
│   ├── ml/
│   │   ├── predict.py         # Prediction logic
│   │   ├── preprocess.py      # Text cleaning
│   │   └── scraper.py         # URL review scraper
│   ├── static/                # Frontend (HTML, CSS, JS)
│   └── main.py                # FastAPI app entry point
├── frontend/                  # React UI (Vite)
├── training/
│   ├── data_pipeline.py       # Load, merge, clean, split datasets
│   ├── train_ml.py            # Train LR, SVM, Random Forest
│   ├── train_dl.py            # Fine-tune DistilBERT
│   ├── compare_models.py      # Evaluate and compare all models
│   └── eda.py                 # Exploratory data analysis
├── datasets/                  # Raw dataset CSVs (not in git)
├── models/                    # Saved model files (lightweight ML artifacts)
├── tests/                     # Unit tests (60 tests)
└── requirements.txt
```

---

## Datasets

| Dataset                     | Size             | Labels                                  | Domain      |
| --------------------------- | ---------------- | --------------------------------------- | ----------- |
| OPUS Deceptive Opinion Spam | ~1,600 reviews   | 4-class (deceptive/truthful × polarity) | Hotels      |
| Amazon Product Reviews      | 40,000+ reviews  | Binary (CG/OR)                          | E-commerce  |
| Yelp (reference)            | 350,000+ reviews | Binary                                  | Restaurants |

Training uses 70/15/15 train/validation/test split with stratification by label and source. Models trained on Amazon+OPUS are also tested on Yelp to evaluate **cross-domain generalization**.

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fake-review-detection.git
cd fake-review-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add datasets

Place the following files in the `datasets/` folder:

- `deceptive-opinion.csv` — OPUS Corpus
- `fake reviews dataset.csv` — Amazon Reviews
- `yelp_data_train.csv` and `yelp_data_test.csv` — Yelp data

### 4. Run the training pipeline

```bash
# Step 1: Process and merge datasets
python -m training.data_pipeline

# Step 2: Train ML models
python -m training.train_ml

# Step 3: (Optional) Fine-tune DistilBERT — requires GPU
python -m training.train_dl

# Step 4: Compare all models
python -m training.compare_models

# Step 5: (Optional) Exploratory data analysis
python -m training.eda
```

### 5. Start the API server

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### 6. (Optional) Run the React frontend (dev)

```bash
cd frontend
npm install
npm run dev
```

The dev server proxies `/api` to FastAPI (port 8000). Visit [http://localhost:5173](http://localhost:5173).

### 7. Build React for production

```bash
cd frontend
npm run build
```

FastAPI will serve `frontend/dist` automatically when it exists.

### Optional configuration

You can override paths and defaults via environment variables:

- `TRUST_REVIEW_BASE_DIR` — project root (defaults to repo root)
- `TRUST_REVIEW_DATASETS_DIR` — datasets directory
- `TRUST_REVIEW_MODELS_DIR` — models directory
- `TRUST_REVIEW_DEFAULT_ML_MODEL` — default ML model name
- `TRUST_REVIEW_DEFAULT_MODEL_TYPE` — `"ml"` or `"dl"`

---

## API Endpoints

| Method | Endpoint             | Description                  |
| ------ | -------------------- | ---------------------------- |
| `POST` | `/api/predict`       | Classify a single review     |
| `POST` | `/api/predict/bulk`  | Bulk classify from CSV/XLSX  |
| `POST` | `/api/predict/url`   | Scrape and classify from URL |
| `GET`  | `/api/download/{id}` | Download labeled CSV         |
| `GET`  | `/api/info`          | List available models        |

### Example Request

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing! Best purchase ever!", "model_type": "ml"}'
```

### Example Response

```json
{
	"result": {
		"text": "this product is absolutely amazing best purchase ever",
		"label": "Fake",
		"confidence": 0.87,
		"model_used": "logistic_regression"
	}
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

**60 unit tests** covering preprocessing, API schemas, data pipeline, model loader, prediction logic, and web scraper.

---

## Model Results

> Run `python -m training.compare_models` after training to generate this table.

| Model                   | Accuracy | Precision | Recall | F1     | ROC-AUC |
| ----------------------- | -------- | --------- | ------ | ------ | ------- |
| 🏆 DistilBERT (fine-tuned) | 90.88%  | 87.94%   | 94.76% | 91.22% | 0.9783  |
| Linear SVM              | 87.44%   | 87.84%    | 86.91% | 87.37% | 0.9490  |
| Logistic Regression     | 86.89%   | 88.36%    | 84.98% | 86.63% | 0.9486  |
| Random Forest           | 83.81%   | 87.42%    | 78.98% | 82.99% | 0.9265  |

DistilBERT trained for 4 epochs on GPU (40 min) with AMP + gradient accumulation (effective batch 32). Best val F1 achieved at epoch 3 (0.9122).

These are v1 numbers. That run picked its best epoch on the test set; retrained with a proper validation split, the same setup scores 90.58%.

### v2: modern LLM fakes (deployed model)

The v1 model caught only **1%** of fakes written by modern LLMs (DeepSeek-v4). v2 adds 10K DeepSeek-v4 fakes, 10K LFM2.5 fakes and 9.9K healthcare reviews (GP Surgery) to training, with source-balanced loss. GLM-5.2 fakes are kept out of training and used only as an unseen-generator test.

| Test slice | v1 | **v2 (deployed)** |
| --- | --- | --- |
| Full v2 test (24,473 reviews) | 60.7% | **93.2%** (AUC 0.988) |
| GLM-5.2 fakes caught (unseen generator) | 0.7% | **98.3%** |
| DeepSeek-v4 / LFM2.5 fakes caught | 1.3% / 1.5% | 99.7% / 99.7% |
| Amazon / OPUS / Yelp (v1 test) | 97.7% / 83.8% / 68.1% | 95.5% / 85.0% / 67.3% |
| v1 test overall | 90.58% | 88.8% |

Known limitations:
- v2 flags about 8% of genuine reviews as fake (Amazon and GLM test sets). It trades some precision on genuine reviews for catching modern LLM fakes.
- GLM fakes used the same prompt protocol as the training fakes, so "unseen" means a new model, not a new generation method.
- Yelp stays near 68% with text alone, because its labels come from Yelp's behaviour-based filter. Reviewer behaviour features (rating deviation, account age, review counts) reach 88.4% when fused with DistilBERT.
- Leave-one-source-out transfer is at chance (about 50%): each dataset's "fake" is a different phenomenon.

The deployed DistilBERT ("dl" model type) is v2; the default "ml" option is still the TF-IDF baseline. Full method, tables and reproduction steps: [experiments/RESULTS.md](https://github.com/ashish6123/Trust-Review/blob/master/experiments/RESULTS.md).

---

## Deployment

The API can be deployed on [Render](https://render.com) with the following start command:

```
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Note: Large model files (DistilBERT and random_forest) are not included in git. Lightweight ML artifacts can be generated locally or bundled for deployment as needed.

---

