"""Shared Stage-3 meta-feature assembly — used identically by train_meta.py and
predict.py so the training and inference feature vectors never drift.

The meta-classifier deliberately consumes the URL submodel's *score* (which
internally encapsulates url_count and the structural URL features), NOT raw
url_count — per the spec's ban on using url_count as a direct meta feature.
``financial_context`` is one-hot encoded so the model can learn that 'invoice'
is roughly neutral while 'bank_alert' is risky (the core false-positive fix).
"""
from __future__ import annotations

import numpy as np

from features.url_features import url_feature_vector

# Order of the meta feature vector. Keep in sync with build_meta_vector().
META_FEATURE_ORDER = (
    "stage1_score", # calibrated Stage-1 phishing probability (0-1)
    "url_score", # URL submodel phishing probability (0 if no URL)
    "urgency_score",
    "credential_request",
    "fin_bank_alert",
    "fin_payment_request",
    "fin_invoice",
    "generic_greeting",
    "threatening_language",
    "package_delivery_context",
    "text_length",
    "html_ratio",
)


def score_url(url_feats: dict, url_bundle: dict) -> float:
    """URL submodel phishing probability for one email. 0.0 when there is no
    URL (no URL → no URL-based suspicion; the cascade relies on other stages)."""
    if not url_feats.get("has_url"):
        return 0.0
    model = url_bundle["model"]
    vec = np.asarray([url_feature_vector(url_feats)], dtype=float)
    idx = list(model.classes_).index(1)
    return float(model.predict_proba(vec)[0][idx])


def build_meta_vector(stage1_score: float, url_score: float,
                      text_feats: dict) -> list[float]:
    """Assemble the ordered numeric meta feature vector."""
    fin = text_feats["financial_context"]
    return [
        float(stage1_score),
        float(url_score),
        float(text_feats["urgency_score"]),
        float(text_feats["credential_request"]),
        1.0 if fin == "bank_alert" else 0.0,
        1.0 if fin == "payment_request" else 0.0,
        1.0 if fin == "invoice" else 0.0,
        float(text_feats["generic_greeting"]),
        float(text_feats["threatening_language"]),
        float(text_feats["package_delivery_context"]),
        float(text_feats["text_length"]),
        float(text_feats["html_ratio"]),
    ]
