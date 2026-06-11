"""Step 5 — Cascade prediction (v2) with risk score and contextual explanations.

predict_email(text) returns:
{
  "verdict": "phishing" | "suspect" | "legitimate",
  "risk_score": 91.4, # 0-100
  "model": "cascade_v2",
  "stage1_score": 88.0, # Stage-1 screening probability (0-100)
  "explanations": [...] # contextual, combination-based
}

Cascade:
  Stage 1 calibrated TF-IDF+LogReg screening (broad recall).
           score < 0.20 -> short-circuit as legitimate.
  Stage 2 URL submodel score + contextual text features.
  Stage 3 calibrated meta-classifier -> final risk score.
  Stage 4 transparent explanations on the extracted features. They JUSTIFY the
           score in human terms; they NEVER override it, and they never cite a
           single signal ("a link is present") on its own — only the
           combinations that actually drove the score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features.text_features import extract_text_features # noqa: E402
from features.url_features import extract_url_features # noqa: E402
from meta_features import build_meta_vector, score_url # noqa: E402
from text_utils import clean_text, find_urls # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "models" / "baseline_tfidf_logreg.joblib"
URL_PATH = ROOT / "models" / "url_classifier.joblib"
META_PATH = ROOT / "models" / "meta_classifier.joblib"

# Cascade thresholds.
STAGE1_SHORTCUT = 0.20 # below this, Stage 1 is confident enough -> legitimate
SUSPECT_THRESHOLD = 0.35
PHISHING_THRESHOLD = 0.60

# Minimum content to make a call. A one-word input like "bonjour" carries no
# signal: the model would otherwise lean on the corpus bias of isolated greeting
# words (which skew phishing) and emit a confident-but-meaningless score. Below
# this, and with no URL to analyze, we honestly return 'indéterminé'.
MIN_WORDS = 3

_CACHE = None


def load_models():
    """Load and cache the three cascade models."""
    global _CACHE
    if _CACHE is None:
        missing = [p for p in (BASELINE_PATH, URL_PATH, META_PATH) if not p.exists()]
        if missing:
            names = ", ".join(p.name for p in missing)
            raise FileNotFoundError(
                f"Modèle(s) introuvable(s): {names}\n"
                "Entraînez le pipeline: python src/train_baseline.py && "
                "python src/train_url_classifier.py && python src/train_meta.py"
            )
        _CACHE = {
            "baseline": joblib.load(BASELINE_PATH),
            "url": joblib.load(URL_PATH),
            "meta": joblib.load(META_PATH),
        }
    return _CACHE


# Backwards-compatible helper (app.py warms the cache / checks for the model).
def load_model(path: Path = BASELINE_PATH):
    return load_models()["baseline"]


def _proba_phishing(model, X) -> float:
    idx = list(model.classes_).index(1)
    return float(model.predict_proba(X)[0][idx])


def build_explanations(url_feats: dict, text_feats: dict, final_score: float) -> list[str]:
    """Contextual, combination-based explanations. Never cites a single signal
    (a link, a payment mention) on its own — only combinations. Read-only."""
    reasons: list[str] = []
    uf, tf = url_feats, text_feats
    has_url = uf["has_url"]
    risky_link = (uf["max_tld_risk_score"] > 0 or uf["has_ip_url"]
                  or uf["domain_typosquatting_score"] >= 0.72)
    official_link = uf["legitimate_domain_ratio"] >= 0.5

    # --- URL-driven combinations ---
    if uf["domain_typosquatting_score"] >= 0.72:
        reasons.append("Typosquatting probable : le domaine du lien imite une marque connue")
    if uf["max_tld_risk_score"] > 0 and uf["has_auth_keyword_in_path"]:
        reasons.append("Lien vers un domaine à TLD à risque pointant vers une page "
                       "de connexion/vérification")
    elif uf["max_tld_risk_score"] > 0:
        reasons.append("Lien vers un domaine à TLD à risque élevé")
    if uf["has_ip_url"]:
        reasons.append("Lien pointant vers une adresse IP brute au lieu d'un domaine")
    if uf["has_redirect_keyword"] and uf["has_auth_keyword_in_path"]:
        reasons.append("Redirection masquée vers une page de connexion")
    if uf["url_text_mismatch"]:
        reasons.append("Le texte affiché du lien diffère du domaine réel de destination")

    # --- text + context combinations ---
    if tf["credential_request"] and (tf["urgency_score"] > 0 or tf["threatening_language"]):
        reasons.append("Demande d'identifiants combinée à une pression temporelle / menace")
    if tf["financial_context"] == "bank_alert" and tf["credential_request"]:
        reasons.append("Alerte de compte bancaire accompagnée d'une demande d'identifiants")
    elif tf["financial_context"] == "bank_alert" and has_url and not official_link:
        reasons.append("Alerte bancaire renvoyant vers un lien qui n'est pas un "
                       "domaine officiel connu")
    if tf["package_delivery_context"] and tf["financial_context"] == "payment_request" \
            and (risky_link or (has_url and not official_link)):
        reasons.append("Prétexte de colis exigeant un paiement via un lien non officiel")
    if tf["generic_greeting"] and (tf["credential_request"] or tf["threatening_language"]):
        reasons.append("Formule d'accueil générique impersonnelle combinée à une "
                       "demande d'action sensible")

    # --- low-risk positive note (honest, not a guarantee) ---
    if not reasons and final_score < SUSPECT_THRESHOLD:
        if official_link:
            reasons.append("Lien(s) vers un ou des domaines connus, sans signal de "
                           "pression ou de demande d'identifiants")
        else:
            reasons.append("Aucun signal de phishing évident détecté")

    if not reasons:
        # Elevated score but no single combination fired: be transparent.
        reasons.append("Score établi par le modèle de texte ; signaux structurels "
                       "individuellement faibles")
    return reasons


def _verdict(score01: float) -> str:
    if score01 > PHISHING_THRESHOLD:
        return "phishing"
    if score01 > SUSPECT_THRESHOLD:
        return "suspect"
    return "legitimate"


def predict_email(text: str, models=None) -> dict:
    """Classify an email through the cascade."""
    if models is None:
        models = load_models()

    cleaned = clean_text(text)
    if not cleaned:
        return {
            "verdict": "legitimate",
            "risk_score": 0.0,
            "model": "cascade_v2",
            "stage1_score": 0.0,
            "explanations": ["texte vide"],
        }

    # --- Guard: not enough content to judge (e.g. a bare "bonjour") ---
    if len(cleaned.split()) < MIN_WORDS and not find_urls(cleaned):
        return {
            "verdict": "indéterminé",
            "risk_score": 0.0,
            "model": "cascade_v2",
            "stage1_score": 0.0,
            "explanations": ["Contenu insuffisant pour une analyse fiable — "
                             "collez le sujet et le corps complet de l'email"],
        }

    # --- Stage 1: screening ---
    stage1 = _proba_phishing(models["baseline"], [cleaned])

    if stage1 < STAGE1_SHORTCUT:
        return {
            "verdict": "legitimate",
            "risk_score": round(stage1 * 100, 1),
            "model": "cascade_v2",
            "stage1_score": round(stage1 * 100, 1),
            "explanations": ["Aucun signal de phishing évident détecté "
                             "(pré-filtre Stage 1)"],
        }

    # --- Stage 2: feature extraction ---
    url_feats = extract_url_features(cleaned)
    text_feats = extract_text_features(text) # raw text for html_ratio
    url_score = score_url(url_feats, models["url"])

    # --- Stage 3: meta score ---
    vec = np.asarray([build_meta_vector(stage1, url_score, text_feats)], dtype=float)
    final = _proba_phishing(models["meta"]["model"], vec)

    # --- Stage 4: explanations (read-only) ---
    explanations = build_explanations(url_feats, text_feats, final)

    return {
        "verdict": _verdict(final),
        "risk_score": round(final * 100, 1),
        "model": "cascade_v2",
        "stage1_score": round(stage1 * 100, 1),
        "explanations": explanations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Détecteur de phishing — prédiction CLI")
    parser.add_argument("--text", help="Texte de l'email à analyser")
    parser.add_argument("--file", help="Chemin d'un fichier texte à analyser")
    args = parser.parse_args()

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8", errors="ignore")
    elif args.text:
        text = args.text
    else:
        text = (
            "URGENT: Your bank account has been blocked. "
            "Verify your password immediately at http://secure-login.tk/verify "
            "to avoid permanent suspension. Dear customer, act now!"
        )
        print(">> Aucun texte fourni — exemple de démonstration utilisé.\n")

    result = predict_email(text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
