# ReturnGuard — AI Return-Risk Checker

ReturnGuard is a Streamlit app that classifies e-commerce return requests to help spot return abuse early. A Random Forest model reads the details of a return and labels it as one of four classes:

| Class | Meaning |
|---|---|
| **Legitimate** | A normal, valid return |
| **Fraudulent Return** | A return made with deceptive intent |
| **Policy Abuser** | A customer who repeatedly exploits return-policy loopholes |
| **Wardrobing** | Buying an item, using it, then returning it |

## Demo

**Try it here:** [Streamlit App](https://return-guard.streamlit.app)

## Blog

**Read the story behind this project:** [Medium Blog](https://medium.com/@your-username/your-blog-post)

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
├── Streamlit/
│   ├── main.py                              # Streamlit app (all pages)
│   ├── evaluate_model.py                    # Computes metrics -> model_metrics.json
│   ├── model_metrics.json                   # Metrics shown on the Model Insight page
│   ├── return_abuse_detector_model.joblib   # Trained RandomForestClassifier
│   └── ecommerce_return_abuse_dataset.csv   # Dataset (60,000 rows)
├── fraud_return_model.ipynb                 # Notebook with the model training code
├── Legacy_main.py                           # Old version of main.py (kept for reference)
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
streamlit run Streamlit/main.py
```

The app opens at `http://localhost:8501`. Keep the `.joblib` model file in the same folder as `main.py` (the `Streamlit/` folder).

> **Note:** `scikit-learn` is pinned to `1.6.1` because that is the version the model was saved with. Other versions can show warnings or produce invalid results when unpickling.

## Model

- **Algorithm:** `RandomForestClassifier` (100 trees, `random_state=42`)
- **Dataset:** 60,000 labeled returns, [E-commerce Return Abuse Detection Dataset](https://huggingface.co/datasets/sarveshchhetri/E-commerce_return_abuse_detection_dataset) on Hugging Face
- **Split:** 80% train / 20% test (48,000 / 12,000), stratified, `random_state=42`
- **Test performance:** accuracy ≈ 99.96%, macro F1 ≈ 0.999

These near-perfect scores most likely reflect how clean and separable the data is, since the dataset used to train the model is synthetic. Real-world performance on messy production data should be expected to be lower.

## Updating the Metrics

To recompute the numbers shown on the Model Insight page:

```bash
cd Streamlit
python evaluate_model.py                      # retrain a clone on 80% and test on 20%
python evaluate_model.py --csv ecommerce_return_abuse_dataset.csv
python evaluate_model.py --use-saved          # evaluate the saved joblib (same split settings required)
```

Without `--csv`, the script downloads the dataset from Hugging Face, which needs `pip install datasets`.

## Limitations

- **Synthetic training data.** The model was trained on a synthetic dataset, not real store transactions. Its patterns are cleaner than real-world fraud, so the reported ~99.96% accuracy should not be read as production performance.
- **Possible overfitting.** The model scores 100% on its own training data, and the metrics page evaluates a retrained clone of the model by default rather than the saved file itself. Treat the numbers as indicative, not as a guarantee.
- **Class imbalance.** About 70% of the data is Legitimate returns, while Fraudulent Return, Policy Abuser, and Wardrobing are much rarer. Minority-class behavior may differ on real data.
- **Decision support only.** A prediction is a risk signal, not proof of fraud. It should not be used to automatically deny refunds or penalize customers without human review.
- **Fairness and bias.** The input features include demographic and location details such as age and country. These can introduce unfair bias if used in real decisions and should be audited first.
- **Fixed features and categories.** The model expects the exact feature set and column order it was trained on, and unseen categories (new countries, payment methods, or product types) may be handled poorly.
- **Session-only history.** The Dashboard and Customer Detail pages only reflect analyses made during the current session. Nothing is saved to a database.

## Credits

* Dataset: Sarvesh Chhetri, [E-commerce Return Abuse Detection Dataset](https://huggingface.co/datasets/sarveshchhetri/E-commerce_return_abuse_detection_dataset) (Hugging Face).
* Icons: [Devicon](https://devicon.dev/) (MIT) and [Simple Icons](https://simpleicons.org/) (CC0). Font: [Inter](https://rsms.me/inter/) (SIL Open Font License).
* Built with: [Streamlit](https://streamlit.io/), [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), and [joblib](https://joblib.readthedocs.io/).
* Author: Kelompok 5, Bunda Mulia University.
