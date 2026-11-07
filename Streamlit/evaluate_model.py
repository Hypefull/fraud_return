"""
evaluate_model.py
-----------------
Menghitung metrik evaluasi model ReturnGuard dan menyimpannya ke model_metrics.json.
File JSON itu dibaca otomatis oleh halaman "Model Insight" di main.py.

Cara pakai (jalankan di folder yang sama dengan main.py):

    pip install datasets scikit-learn pandas joblib
    python evaluate_model.py                       # ambil dataset dari Hugging Face
    python evaluate_model.py --csv dataset.csv     # atau pakai CSV lokal
    python evaluate_model.py --random-state 42     # samakan dengan notebook trainingmu!

PENTING:
- Default-nya script MELATIH ULANG model dengan hyperparameter yang sama persis
  (clone dari model tersimpan) di 80% data, lalu menguji di 20% sisanya.
  Hasilnya bersih dari kebocoran data, tapi itu model "kembaran", bukan file joblib-nya.
- Pakai --use-saved kalau mau mengevaluasi model joblib yang sudah ada. Hasilnya HANYA valid
  kalau --random-state dan --test-size sama persis dengan saat kamu training dulu.
  Kalau beda, sebagian data uji ternyata sudah pernah dilihat model (data leakage)
  dan skornya jadi terlalu bagus.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import label_binarize

HF_NAME = "sarveshchhetri/E-commerce_return_abuse_detection_dataset"
TARGET = "abuse_type"
NEGATIVE_CLASS = "Legitimate"
HERE = Path(__file__).parent


def load_data(csv_path):
    if csv_path:
        return pd.read_csv(csv_path)
    from datasets import load_dataset

    ds = load_dataset(HF_NAME)
    split = "train" if "train" in ds else list(ds.keys())[0]
    return ds[split].to_pandas()


def f(x):
    return float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--model", default=str(HERE / "return_abuse_detector_model.joblib"))
    ap.add_argument("--out", default=str(HERE / "model_metrics.json"))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--use-saved", action="store_true")
    args = ap.parse_args()

    saved = joblib.load(args.model)
    features = list(saved.feature_names_in_)

    df = load_data(args.csv)
    missing = [c for c in features + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Kolom ini tidak ada di dataset: {missing}\n"
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    X = df[features]
    y = df[TARGET].astype(str)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    est = saved if args.use_saved else clone(saved).fit(X_tr, y_tr)
    classes = [str(c) for c in est.classes_]

    pred = est.predict(X_te)
    proba = est.predict_proba(X_te)
    pred_tr = est.predict(X_tr)

    # ---------- metrik keseluruhan ----------
    majority = y_tr.value_counts(normalize=True).max()
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    overall = {
        "accuracy": f(accuracy_score(y_te, pred)),
        "balanced_accuracy": f(balanced_accuracy_score(y_te, pred)),
        "macro_precision": f(p_m),
        "macro_recall": f(r_m),
        "macro_f1": f(f_m),
        "weighted_f1": f(f1_score(y_te, pred, average="weighted", zero_division=0)),
        "mcc": f(matthews_corrcoef(y_te, pred)),
        "log_loss": f(log_loss(y_te, proba, labels=classes)),
        "roc_auc_ovr_macro": f(roc_auc_score(y_te, proba, multi_class="ovr", average="macro", labels=classes)),
        "majority_baseline_accuracy": f(majority),
        "train_accuracy": f(accuracy_score(y_tr, pred_tr)),
        "train_macro_f1": f(f1_score(y_tr, pred_tr, average="macro", zero_division=0)),
    }

    # ---------- per kelas ----------
    p, r, f1, sup = precision_recall_fscore_support(y_te, pred, labels=classes, zero_division=0)
    y_bin = label_binarize(y_te, classes=classes)
    per_class = {}
    for i, c in enumerate(classes):
        per_class[c] = {
            "precision": f(p[i]),
            "recall": f(r[i]),
            "f1": f(f1[i]),
            "support": int(sup[i]),
            "roc_auc": f(roc_auc_score(y_bin[:, i], proba[:, i])),
            "avg_precision": f(average_precision_score(y_bin[:, i], proba[:, i])),
        }

    # ---------- confusion matrix ----------
    cm = confusion_matrix(y_te, pred, labels=classes)

    # ---------- tampilan biner: wajar vs tidak wajar ----------
    yt = (y_te != NEGATIVE_CLASS).to_numpy()
    yp = (pd.Series(pred, index=y_te.index) != NEGATIVE_CLASS).to_numpy()
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[False, True]).ravel()
    legit_idx = classes.index(NEGATIVE_CLASS) if NEGATIVE_CLASS in classes else None
    score_flag = 1 - proba[:, legit_idx] if legit_idx is not None else None
    binary = {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": f(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": f(tp / (tp + fn)) if (tp + fn) else 0.0,
        "false_positive_rate": f(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": f(fn / (fn + tp)) if (fn + tp) else 0.0,
        "roc_auc": f(roc_auc_score(yt, score_flag)) if score_flag is not None else None,
    }
    binary["f1"] = (
        f(2 * binary["precision"] * binary["recall"] / (binary["precision"] + binary["recall"]))
        if (binary["precision"] + binary["recall"]) else 0.0
    )

    # ---------- cross-validation (hanya di 80% data latih) ----------
    cv = cross_validate(
        clone(saved), X_tr, y_tr,
        cv=StratifiedKFold(args.cv_folds, shuffle=True, random_state=args.random_state),
        scoring=["accuracy", "f1_macro"], n_jobs=-1,
    )
    cv_res = {
        "folds": args.cv_folds,
        "accuracy_mean": f(cv["test_accuracy"].mean()), "accuracy_std": f(cv["test_accuracy"].std()),
        "macro_f1_mean": f(cv["test_f1_macro"].mean()), "macro_f1_std": f(cv["test_f1_macro"].std()),
    }

    # ---------- permutation importance (di data uji) ----------
    pi = permutation_importance(
        est, X_te, y_te, scoring="f1_macro", n_repeats=5,
        random_state=args.random_state, n_jobs=-1,
    )
    perm = {n: {"mean": f(m), "std": f(s)} for n, m, s in zip(features, pi.importances_mean, pi.importances_std)}

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "mode": "saved_model" if args.use_saved else "retrain_clone",
            "dataset": args.csv or HF_NAME,
        },
        "dataset": {
            "n_rows": int(len(df)),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "stratified": True,
            "class_counts_train": {k: int(v) for k, v in y_tr.value_counts().items()},
            "class_counts_test": {k: int(v) for k, v in y_te.value_counts().items()},
        },
        "model": {"type": type(saved).__name__, "params": saved.get_params()},
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
        "binary": binary,
        "cv": cv_res,
        "permutation_importance": perm,
    }

    Path(args.out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Selesai. Metrik disimpan ke {args.out}")
    print(f"Accuracy {overall['accuracy']:.4f} | Macro-F1 {overall['macro_f1']:.4f} | "
          f"Baseline mayoritas {majority:.4f}")


if __name__ == "__main__":
    main()