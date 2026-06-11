"""Step 3 — Baseline model: TF-IDF + Logistic Regression.

- stratified train/test split (never evaluate on training data)
- saves the fitted pipeline to models/baseline_tfidf_logreg.joblib
- saves metrics to reports/baseline_metrics.json
- saves the confusion matrix figure to reports/baseline_confusion_matrix.png

Priority metric: PHISHING RECALL (we care most about catching phishing).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "processed" / "emails.csv"
MODEL_PATH = ROOT / "models" / "baseline_tfidf_logreg.joblib"
METRICS_PATH = ROOT / "reports" / "baseline_metrics.json"
CM_PATH = ROOT / "reports" / "baseline_confusion_matrix.png"

RANDOM_STATE = 42

# Stage-1 is a broad screening net: we lower the decision threshold from 0.5 to
# 0.35 so few real phishing emails slip through. The downstream meta-classifier
# (Stage 3) is responsible for precision; Stage 1 only needs high recall.
STAGE1_THRESHOLD = 0.35


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
    )


def _logreg() -> LogisticRegression:
    """class_weight='balanced' nudges the model toward catching the minority
    class, which helps phishing recall."""
    return LogisticRegression(max_iter=1000, C=5.0, class_weight="balanced")


def build_pipeline() -> Pipeline:
    """TF-IDF + calibrated Logistic Regression.

    CalibratedClassifierCV(cv=5) wraps the LogReg so the probabilities we expose
    via predict_proba are well-calibrated — the cascade uses those probabilities
    directly (not raw decision scores) for the final risk score.
    """
    return Pipeline([
        ("tfidf", _tfidf()),
        ("clf", CalibratedClassifierCV(_logreg(), cv=5)),
    ])


def build_uncalibrated_pipeline() -> Pipeline:
    """Same features + raw LogReg — used only to log the before/after-calibration
    comparison; not saved."""
    return Pipeline([("tfidf", _tfidf()), ("clf", _logreg())])


def _phishing_proba(pipe: Pipeline, X) -> "list[float]":
    classes = list(pipe.classes_)
    idx = classes.index(1) if 1 in classes else len(classes) - 1
    return pipe.predict_proba(X)[:, idx]


def save_confusion_matrix(cm, path: Path) -> None:
    """Render the confusion matrix to a PNG (best effort)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["legit", "phishing"],
            yticklabels=["legit", "phishing"], ax=ax,
        )
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title("Confusion matrix — baseline")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f" [OK] Matrice de confusion: {path}")
    except Exception as exc: # noqa: BLE001
        print(f" ! Figure non générée ({exc}) — on continue.")


def main() -> int:
    if not CSV.exists():
        print(f"! Fichier introuvable: {CSV}")
        print(" Lancez d'abord: python src/download_dataset.py")
        return 1

    df = pd.read_csv(CSV)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip().str.len() > 0]
    df["label"] = df["label"].astype(int)

    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)} (split stratifié)")

    # --- before calibration: raw LogReg, for the comparison log ---
    print("Entraînement (avant calibration) TF-IDF + LogisticRegression...")
    raw_pipe = build_uncalibrated_pipeline()
    raw_pipe.fit(X_train, y_train)
    raw_proba = _phishing_proba(raw_pipe, X_test)
    raw_brier = brier_score_loss(y_test, raw_proba)
    raw_recall_035 = recall_score(y_test, (raw_proba >= STAGE1_THRESHOLD).astype(int),
                                  pos_label=1, zero_division=0)

    # --- after calibration: this is the saved Stage-1 model ---
    print("Entraînement (après calibration) CalibratedClassifierCV(cv=5)...")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    proba = _phishing_proba(pipe, X_test)
    brier = brier_score_loss(y_test, proba)

    # Stage-1 verdict uses the lowered 0.35 threshold (broad net).
    y_pred = (proba >= STAGE1_THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    # pos_label=1 -> phishing
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("=" * 60)
    print(" CALIBRATION — avant / après (Brier plus bas = mieux calibré)")
    print("=" * 60)
    print(f" Brier score avant ..... {raw_brier:.4f}")
    print(f" Brier score après ..... {brier:.4f}")
    print(f" Recall@0.35 avant ..... {raw_recall_035:.4f}")
    print(f" Recall@0.35 après ..... {recall:.4f}")
    print("=" * 60)
    print(f" RÉSULTATS BASELINE STAGE-1 (seuil={STAGE1_THRESHOLD}, positif=phishing)")
    print("=" * 60)
    print(f" Accuracy ............... {accuracy:.4f}")
    print(f" Precision (phishing) ... {precision:.4f}")
    print(f" Recall (phishing) ...... {recall:.4f} <-- prioritaire (>= 0.95 visé)")
    print(f" F1-score (phishing) .... {f1:.4f}")
    print(" Confusion matrix [lignes=réel, colonnes=prédit]:")
    print(f" prédit_legit prédit_phish")
    print(f" réel_legit {cm[0,0]:>8} {cm[0,1]:>8}")
    print(f" réel_phish {cm[1,0]:>8} {cm[1,1]:>8}")
    print("-" * 60)
    print(classification_report(
        y_test, y_pred, target_names=["legitimate", "phishing"],
        zero_division=0,
    ))

    # --- persist artifacts ---
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"[OK] Modèle sauvegardé: {MODEL_PATH}")

    metrics = {
        "model": "baseline_tfidf_logreg",
        "positive_class": "phishing (1)",
        "stage1_threshold": STAGE1_THRESHOLD,
        "calibrated": True,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": round(float(accuracy), 4),
        "precision_phishing": round(float(precision), 4),
        "recall_phishing": round(float(recall), 4),
        "f1_phishing": round(float(f1), 4),
        "calibration": {
            "brier_before": round(float(raw_brier), 4),
            "brier_after": round(float(brier), 4),
            "recall_at_threshold_before": round(float(raw_recall_035), 4),
            "recall_at_threshold_after": round(float(recall), 4),
        },
        "confusion_matrix": {
            "labels": ["legitimate", "phishing"],
            "matrix": cm.tolist(),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"[OK] Métriques sauvegardées: {METRICS_PATH}")

    save_confusion_matrix(cm, CM_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
