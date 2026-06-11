"""Stage 2 submodel — URL classifier.

Trains a RandomForest on the *structured URL features* (not raw URLs) extracted
from emails that contain at least one URL. The email-level label is used as the
target: there are no per-URL labels in the corpus, so a URL-bearing email's
label is the best available signal. The point of this submodel is to learn
which *combinations* of URL features (suspicious TLD + auth keyword + lookalike
domain, etc.) indicate phishing — never URL presence alone.

Output: models/url_classifier.joblib (dict: model + feature order)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features.url_features import ( # noqa: E402
    URL_FEATURE_ORDER,
    extract_url_features,
    url_feature_vector,
)

CSV = ROOT / "data" / "processed" / "emails.csv"
MODEL_PATH = ROOT / "models" / "url_classifier.joblib"
METRICS_PATH = ROOT / "reports" / "url_classifier_metrics.json"
RANDOM_STATE = 42

# Probes that must hold for the submodel to be considered correct.
PROBE_LEGIT = "https://www.amazon.fr/order/123"
PROBE_PHISH = "http://amaz0n-secure.tk/login?redirect=verify"


def build_matrix(df: pd.DataFrame) -> "tuple[np.ndarray, np.ndarray, int]":
    """Extract URL features for URL-bearing emails only."""
    rows, labels = [], []
    skipped = 0
    for text, label in zip(df["text"].values, df["label"].values):
        feats = extract_url_features(text)
        if not feats["has_url"]:
            skipped += 1
            continue
        rows.append(url_feature_vector(feats))
        labels.append(int(label))
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), skipped


def main() -> int:
    if not CSV.exists():
        print(f"! Fichier introuvable: {CSV}")
        return 1

    df = pd.read_csv(CSV)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    # Share the SAME split as the baseline (same seed/test_size/stratify) so the
    # meta-classifier can later consume out-of-sample URL scores on the baseline
    # test rows without leakage. We train on URL-bearing rows of the train
    # portion and evaluate on URL-bearing rows of the test portion.
    df_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE,
    )
    print("Extraction des features URL (emails contenant >=1 URL)...")
    X_train, y_train, sk_tr = build_matrix(df_train)
    X_test, y_test, sk_te = build_matrix(df_test)
    print(f" train: {len(X_train)} avec URL ({sk_tr} sans) | "
          f"test: {len(X_test)} avec URL ({sk_te} sans)")
    print(f" train répartition: phishing={int(y_train.sum())} | "
          f"legit={int((y_train == 0).sum())}")

    clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("Entraînement RandomForest(n_estimators=100, class_weight='balanced')...")
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, list(clf.classes_).index(1)]
    y_pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

    print("=" * 60)
    print(" URL CLASSIFIER (classe positive = phishing)")
    print("=" * 60)
    print(f" AUC .................... {auc:.4f} (>= 0.92 visé)")
    print(f" Precision (phishing) ... {precision:.4f}")
    print(f" Recall (phishing) ...... {recall:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["legitimate", "phishing"],
                                zero_division=0))

    # Feature importances — sanity check that no single feature dominates absurdly.
    print(" Importances des features:")
    for name, imp in sorted(zip(URL_FEATURE_ORDER, clf.feature_importances_),
                            key=lambda t: -t[1]):
        print(f" {name:<28} {imp:.3f}")

    # --- probes ---
    def score(text: str) -> float:
        f = extract_url_features(text)
        v = np.asarray([url_feature_vector(f)], dtype=float)
        return float(clf.predict_proba(v)[0][list(clf.classes_).index(1)])

    legit_score = score(PROBE_LEGIT)
    phish_score = score(PROBE_PHISH)
    print("-" * 60)
    print(f" PROBE legit {PROBE_LEGIT}\n -> {legit_score:.3f} (<= 0.15 visé)")
    print(f" PROBE phish {PROBE_PHISH}\n -> {phish_score:.3f} (>= 0.85 visé)")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_order": list(URL_FEATURE_ORDER)}, MODEL_PATH)
    print(f"[OK] Modèle sauvegardé: {MODEL_PATH}")

    METRICS_PATH.write_text(json.dumps({
        "model": "url_classifier_randomforest",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "auc": round(float(auc), 4),
        "precision_phishing": round(float(precision), 4),
        "recall_phishing": round(float(recall), 4),
        "feature_importances": {
            n: round(float(i), 4)
            for n, i in zip(URL_FEATURE_ORDER, clf.feature_importances_)
        },
        "probes": {
            "legit_amazon": round(legit_score, 3),
            "phish_amaz0n_tk": round(phish_score, 3),
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Métriques sauvegardées: {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
