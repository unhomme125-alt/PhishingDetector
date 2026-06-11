"""Stage 2 — contextual text feature extraction (extraction only, no model).

Critical rule from the spec: an invoice/payment mention is NOT phishing by
itself. ``financial_context == 'invoice'`` alone is benign; it is the
combination (e.g. ``bank_alert`` + ``credential_request``) that the
meta-classifier learns to treat as risky. This module only measures; it does
not decide.
"""
from __future__ import annotations

import re

# --- urgency / pressure ---
_URGENCY = (
    "urgent", "immediately", "as soon as possible", "asap", "act now",
    "right away", "expire", "expires", "expiring", "last chance",
    "within 24 hours", "within 24h", "24 hours", "limited time", "final notice",
    "immédiat", "dès que possible", "expire", "dernier avertissement",
    "sous 24h", "dès maintenant",
)

# --- explicit credential / identity requests ---
_CREDENTIAL = (
    "your password", "enter your password", "confirm your password",
    "verify your account", "verify your identity", "confirm your identity",
    "update your account", "log in to verify", "sign in to verify",
    "enter your credentials", "your username and password", "your pin",
    "social security", "ssn",
    "mot de passe", "vérifiez votre compte", "confirmez votre identité",
    "vos identifiants", "code confidentiel",
)

# --- financial context buckets (order = priority) ---
_BANK_ALERT = (
    "account suspended", "account blocked", "account locked",
    "account will be suspended", "account has been blocked",
    "unusual activity", "suspicious activity", "unauthorized access",
    "account has been limited", "restore your account",
    "compte suspendu", "compte bloqué", "activité inhabituelle",
    "compte limité",
)
_PAYMENT_REQUEST = (
    "pay now", "payment required", "customs fee", "outstanding payment",
    "wire transfer", "transfer funds", "release your", "to release",
    "settle your", "overdue payment",
    "payez maintenant", "frais de douane", "paiement requis", "virement",
)
_INVOICE = (
    "invoice", "receipt", "your statement", "statement is ready",
    "billing statement", "order confirmation", "payment confirmation",
    "facture", "reçu", "relevé", "confirmation de commande",
)

# --- generic greetings ---
_GENERIC = (
    "dear customer", "dear user", "dear member", "dear client",
    "valued customer", "dear account holder", "dear sir/madam",
    "cher client", "cher utilisateur", "madame, monsieur",
)

# --- threatening language ---
_THREAT = (
    "will be suspended", "will be closed", "will be terminated",
    "will be deleted", "permanently suspended", "legal action",
    "permanently closed", "lose access", "account closure",
    "sera suspendu", "sera fermé", "sera supprimé", "poursuites",
)

# --- package / delivery ---
_DELIVERY = (
    "package", "parcel", "delivery", "shipment", "tracking", "customs",
    "could not be delivered", "delivery attempt",
    "colis", "livraison", "suivi", "douane", "n'a pas pu être livré",
)

_TAG_RE = re.compile(r"<[^>]+>")


def _hits(low: str, needles) -> int:
    return sum(1 for n in needles if n in low)


def _contains(low: str, needles) -> bool:
    return any(n in low for n in needles)


def extract_text_features(text: str) -> dict:
    """Extract contextual text features. See module docstring for the
    invoice-is-not-phishing rule."""
    raw = text or ""
    low = raw.lower()

    # html_ratio computed on the raw input (before any stripping).
    tag_chars = sum(len(m) for m in _TAG_RE.findall(raw))
    html_ratio = round(tag_chars / len(raw), 3) if raw else 0.0

    urgency_hits = _hits(low, _URGENCY)
    urgency_score = round(min(1.0, urgency_hits / 3.0), 3)

    # Financial context — single bucket, highest-risk wins.
    if _contains(low, _BANK_ALERT):
        financial = "bank_alert"
    elif _contains(low, _PAYMENT_REQUEST):
        financial = "payment_request"
    elif _contains(low, _INVOICE):
        financial = "invoice"
    else:
        financial = "none"

    return {
        "urgency_score": urgency_score,
        "credential_request": _contains(low, _CREDENTIAL),
        "financial_context": financial,
        "generic_greeting": _contains(low, _GENERIC),
        "threatening_language": _contains(low, _THREAT),
        "package_delivery_context": _contains(low, _DELIVERY),
        "text_length": len(raw),
        "html_ratio": html_ratio,
    }
