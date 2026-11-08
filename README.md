# ReturnGuard — AI Return-Risk Checker

ReturnGuard is a Streamlit app that classifies e-commerce return requests to help spot return abuse early. A Random Forest model reads the details of a return and labels it as one of four classes:

| Class | Meaning |
|---|---|
| **Legitimate** | A normal, valid return |
| **Fraudulent Return** | A return made with deceptive intent |
| **Policy Abuser** | A customer who repeatedly exploits return-policy loopholes |
| **Wardrobing** | Buying an item, using it, then returning it |

## Features

- **Home** — project overview and tech stack.
- **Dashboard** — summary of the analyses run in the current session
- **Prediction** — enter the customer and return details and get an instant risk classification
- **Customer Detail** — inspect the features and data behind an individual analysis
- **Model Insight** — dataset split, performance metrics, confusion matrix, cross-validation, and permutation importance, read from `model_metrics.json`

## Tech Stack

Python · Streamlit · scikit-learn · pandas · joblib

## Project Structure

```
.
├── main.py                                  # Streamlit app (all pages)
├── evaluate_model.py                        # Computes metrics -> model_metrics.json
├── model_metrics.json                       # Metrics shown on the Model Insight page
├── return_abuse_detector_model.joblib       # Trained RandomForestClassifier
├── ecommerce_return_abuse_dataset.csv       # Dataset (60,000 rows)
├── requirements.txt
└── README.md
```

## Getting Started

```bash
# 1. Clone
git clone https://github.com/Hypefull/fraud_return.git
cd fraud_return

# 2. (Recommended) virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run main.py
```

The app opens at `http://localhost:8501`. Keep the `.joblib` model file in the same folder as `main.py`.

> **Note:** `scikit-learn` is pinned to `1.6.1` because that is the version the model was saved with. Other versions can show warnings or produce invalid results when unpickling.

## Model

- **Algorithm:** `RandomForestClassifier` (100 trees, `random_state=42`)
- **Dataset:** 60,000 labeled returns, [E-commerce Return Abuse Detection Dataset](https://huggingface.co/datasets/sarveshchhetri/E-commerce_return_abuse_detection_dataset) on Hugging Face
- **Split:** 80% train / 20% test (48,000 / 12,000), stratified, `random_state=42`
- **Test performance:** accuracy ≈ 99.96%, macro F1 ≈ 0.999

These near-perfect scores most likely reflect how clean and separable the dataset due to the fact the dataset used to train the model is synthetic, so real-world performance on messy production data should be expected to be lower.

## Updating the Metrics

To recompute the numbers shown on the Model Insight page:

```bash
python evaluate_model.py                      # retrain a clone on 80% and test on 20%
python evaluate_model.py --csv ecommerce_return_abuse_dataset.csv
python evaluate_model.py --use-saved          # evaluate the saved joblib (same split settings required)
```

Without `--csv`, the script downloads the dataset from Hugging Face, which needs `pip install datasets`.

## Limitations


