"""Stage 3 — meta-classifier training.

Combines the Stage-1 calibrated text score with the Stage-2 outputs (URL submodel
score + contextual text features) into a final calibrated risk score.

NO LEAKAGE: the meta is trained only on the baseline TEST split (same seed as
train_baseline / train_url_classifier), so the Stage-1 and URL scores it learns
from are genuine out-of-sample predictions. That split is then divided into
meta-train / meta-test for honest evaluation.

Model: LogisticRegression (with StandardScaler) wrapped in
CalibratedClassifierCV(method='isotonic'). Coefficients are logged so we can
confirm no single feature dominates absurdly.

Output: models/meta_classifier.joblib (dict: model + feature order)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features.text_features import extract_text_features # noqa: E402
from features.url_features import extract_url_features # noqa: E402
from meta_features import META_FEATURE_ORDER, build_meta_vector, score_url # noqa: E402
from text_utils import clean_text # noqa: E402

CSV = ROOT / "data" / "processed" / "emails.csv"
BASELINE_PATH = ROOT / "models" / "baseline_tfidf_logreg.joblib"
URL_PATH = ROOT / "models" / "url_classifier.joblib"
MODEL_PATH = ROOT / "models" / "meta_classifier.joblib"
METRICS_PATH = ROOT / "reports" / "meta_metrics.json"
RANDOM_STATE = 42


def _stage1_proba(baseline, texts) -> np.ndarray:
    cleaned = [clean_text(t) for t in texts]
    idx = list(baseline.classes_).index(1)
    return baseline.predict_proba(cleaned)[:, idx]


def build_meta_matrix(texts, baseline, url_bundle) -> np.ndarray:
    s1 = _stage1_proba(baseline, texts)
    rows = []
    for text, stage1 in zip(texts, s1):
        uf = extract_url_features(text)
        tf = extract_text_features(text)
        rows.append(build_meta_vector(stage1, score_url(uf, url_bundle), tf))
    return np.asarray(rows, dtype=float)


def main() -> int:
    for p in (CSV, BASELINE_PATH, URL_PATH):
        if not p.exists():
            print(f"! Fichier introuvable: {p}")
            return 1

    df = pd.read_csv(CSV)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    # Same split as the other stages → baseline TEST rows are out-of-sample.
    _, df_test = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE,
    )
    print(f"Rows disponibles pour le meta (test baseline): {len(df_test)}")

    baseline = joblib.load(BASELINE_PATH)
    url_bundle = joblib.load(URL_PATH)

    print("Construction des features meta (stage1 + url + texte)...")
    X = build_meta_matrix(df_test["text"].values, baseline, url_bundle)
    y = df_test["label"].values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE,
    )

    base = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000,
                                                class_weight="balanced"))])
    meta = CalibratedClassifierCV(base, method="isotonic", cv=5)
    print("Entraînement meta: LogReg + CalibratedClassifierCV(isotonic)...")
    meta.fit(Xtr, ytr)

    idx = list(meta.classes_).index(1)
    proba = meta.predict_proba(Xte)[:, idx]
    auc = roc_auc_score(yte, proba)

    print("=" * 60)
    print(" META-CLASSIFIER (classe positive = phishing)")
    print("=" * 60)
    print(f" AUC .................... {auc:.4f}")
    for thr in (0.50, 0.60):
        pred = (proba >= thr).astype(int)
        prec = precision_score(yte, pred, pos_label=1, zero_division=0)
        rec = recall_score(yte, pred, pos_label=1, zero_division=0)
        flag = " <-- seuil verdict phishing" if thr == 0.60 else ""
        print(f" @ seuil {thr:.2f}: precision={prec:.4f} recall={rec:.4f}{flag}")
        if thr == 0.50:
            prec50, rec50 = prec, rec

    pred50 = (proba >= 0.50).astype(int)
    print(classification_report(yte, pred50,
                                target_names=["legitimate", "phishing"],
                                zero_division=0))

    # --- coefficient sanity check (fit a plain, inspectable LogReg) ---
    insp = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000,
                                                class_weight="balanced"))])
    insp.fit(Xtr, ytr)
    coefs = insp.named_steps["clf"].coef_[0]
    print(" Coefficients meta (sur features standardisées):")
    for name, c in sorted(zip(META_FEATURE_ORDER, coefs), key=lambda t: -abs(t[1])):
        print(f" {name:<26} {c:+.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": meta, "feature_order": list(META_FEATURE_ORDER)}, MODEL_PATH)
    print(f"[OK] Modèle sauvegardé: {MODEL_PATH}")

    METRICS_PATH.write_text(json.dumps({
        "model": "meta_logreg_isotonic",
        "n_meta_train": int(len(Xtr)),
        "n_meta_test": int(len(Xte)),
        "auc": round(float(auc), 4),
        "precision_phishing_at_0.50": round(float(prec50), 4),
        "recall_phishing_at_0.50": round(float(rec50), 4),
        "coefficients": {n: round(float(c), 4)
                         for n, c in zip(META_FEATURE_ORDER, coefs)},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Métriques sauvegardées: {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
