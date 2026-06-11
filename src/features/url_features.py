"""Stage 2 — URL feature extraction (extraction only, no model).

Turns the URLs found in an email into a small structured feature dict. The
guiding rule of the v2 spec: the *presence* of a URL is never a
phishing signal on its own. What matters is the combination of structural
features below, which a trained model weighs.

The BRAND reference list here is a *feature input* (for typosquatting similarity
and a known-domain ratio), NOT a verdict whitelist. It is never used to override
a score — it only produces numbers that feed the meta-classifier.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from urllib.parse import urlparse

import tldextract

# tldextract normally hits the network once to refresh the public-suffix list.
# Pin to the bundled snapshot so feature extraction stays offline + deterministic.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

# Suspicious TLDs (cheap, frequently-abused, free/low-cost). Extend as needed.
SUSPICIOUS_TLDS = frozenset({
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "click",
    "download", "work", "party",
})

# Top legitimate brand domain labels — reference for typosquatting similarity
# and the known-domain ratio. A feature input, never a hard override.
BRANDS = frozenset({
    "amazon", "google", "paypal", "microsoft", "apple", "facebook", "netflix",
    "linkedin", "github", "dropbox", "stripe", "shopify", "instagram", "twitter",
    "youtube", "outlook", "office365", "icloud", "adobe", "spotify", "ebay",
    "chronopost", "laposte", "labanquepostale", "orange", "sfr", "free",
    "bnpparibas", "creditmutuel", "zoom", "slack", "docusign", "wetransfer",
    "fedex", "ups", "dhl", "colissimo", "ameli", "impots", "booking",
    "airbnb", "uber", "whatsapp", "yahoo", "gmail", "protonmail", "salesforce",
    "wellsfargo", "chase", "bankofamerica",
})

# Path/query tokens.
_REDIRECT_KEYWORDS = ("redirect", "url=", "link=", "goto=", "return=", "next=")
_AUTH_KEYWORDS = ("login", "log-in", "signin", "sign-in", "verify", "secure",
                   "account", "password", "credential", "confirm", "update")

# Normal-looking suffixes used when deciding the known-domain ratio.
_COMMON_TLDS = frozenset({
    "com", "org", "net", "edu", "gov", "io", "co", "fr", "de", "uk", "us",
    "ca", "es", "it", "nl", "be", "ch", "eu",
})

from text_utils import find_urls # noqa: E402 (src is on sys.path at import time)


# Fixed feature order shared by the URL classifier, the meta-classifier and
# predict.py so every vector is assembled identically. 'has_url' is omitted on
# purpose (the URL classifier only sees URL-bearing emails; downstream stages
# gate on it separately).
URL_FEATURE_ORDER = (
    "url_count",
    "max_tld_risk_score",
    "has_ip_url",
    "max_subdomain_depth",
    "has_redirect_keyword",
    "has_auth_keyword_in_path",
    "domain_typosquatting_score",
    "url_text_mismatch",
    "legitimate_domain_ratio",
)


def url_feature_vector(feats: dict) -> list[float]:
    """Turn an extract_url_features() dict into an ordered numeric vector."""
    return [float(feats[k]) for k in URL_FEATURE_ORDER]


def _normalize(url: str) -> str:
    """Ensure the URL has a scheme so urlparse/tldextract behave."""
    return url if "://" in url else "http://" + url


def _is_ip_host(host: str) -> bool:
    parts = host.split(".")
    return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


def _subdomain_depth(subdomain: str) -> int:
    """Number of subdomain labels, ignoring a leading 'www'."""
    if not subdomain:
        return 0
    labels = [l for l in subdomain.split(".") if l and l != "www"]
    return len(labels)


def _tokens(label: str) -> list[str]:
    """Split a domain label into alpha/digit-ish tokens for lookalike checks."""
    out, cur = [], []
    for ch in label:
        if ch in "-_.":
            if cur:
                out.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out or [label]


def _typosquat_score(domain_label: str) -> float:
    """0.0 if the label is an exact brand (real spelling) or unrelated;
    high (close to 1.0) if it looks like a misspelled / decorated brand."""
    if not domain_label or domain_label in BRANDS:
        return 0.0
    best = 0.0
    candidates = [domain_label] + _tokens(domain_label)
    for cand in candidates:
        if cand in BRANDS: # exact brand token embedded → handled elsewhere
            continue
        for brand in BRANDS:
            best = max(best, SequenceMatcher(None, cand, brand).ratio())
    # Only report genuinely close lookalikes; ignore incidental similarity.
    return round(best, 3) if best >= 0.72 else 0.0


def extract_url_features(text: str) -> dict:
    """Extract structured URL features from all URLs in ``text``.

    See module docstring: presence of a URL is never itself a phishing signal.
    """
    urls = find_urls(text or "")
    if not urls:
        return {
            "has_url": False,
            "url_count": 0,
            "max_tld_risk_score": 0.0,
            "has_ip_url": False,
            "max_subdomain_depth": 0,
            "has_redirect_keyword": False,
            "has_auth_keyword_in_path": False,
            "domain_typosquatting_score": 0.0,
            "url_text_mismatch": False,
            "legitimate_domain_ratio": 0.0,
        }

    max_tld_risk = 0.0
    has_ip = False
    max_depth = 0
    has_redirect = False
    has_auth = False
    max_typo = 0.0
    legit_hits = 0

    for raw in urls:
        url = _normalize(raw.rstrip(".,);]>\"'"))
        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            ext = _EXTRACT(url)
            path_q = (parsed.path + "?" + parsed.query).lower()
        except ValueError:
            # Malformed URL (e.g. bad IPv6 brackets) — still counts as a URL,
            # but we can't extract structure from it.
            continue

        if _is_ip_host(host):
            has_ip = True

        suffix = ext.suffix.lower()
        last_tld = suffix.rsplit(".", 1)[-1] if suffix else ""
        if last_tld in SUSPICIOUS_TLDS:
            max_tld_risk = 1.0

        max_depth = max(max_depth, _subdomain_depth(ext.subdomain.lower()))

        if any(k in path_q for k in _REDIRECT_KEYWORDS):
            has_redirect = True
        if any(k in path_q for k in _AUTH_KEYWORDS):
            has_auth = True

        label = ext.domain.lower()
        max_typo = max(max_typo, _typosquat_score(label))

        if label in BRANDS and last_tld in _COMMON_TLDS:
            legit_hits += 1

    return {
        "has_url": True,
        "url_count": len(urls),
        "max_tld_risk_score": float(max_tld_risk),
        "has_ip_url": has_ip,
        "max_subdomain_depth": int(max_depth),
        "has_redirect_keyword": has_redirect,
        "has_auth_keyword_in_path": has_auth,
        "domain_typosquatting_score": float(max_typo),
        "url_text_mismatch": _anchor_mismatch(text),
        "legitimate_domain_ratio": round(legit_hits / len(urls), 3),
    }


def _anchor_mismatch(text: str) -> bool:
    """Best-effort: True if an <a> anchor text names a domain different from its
    href's real domain. Returns False when there is no HTML anchor (the common
    case for this plaintext corpus)."""
    if not text or "<a" not in text.lower():
        return False
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return False
    try:
        soup = BeautifulSoup(text, "html.parser")
    except Exception:
        return False
    for a in soup.find_all("a", href=True):
        href_dom = _EXTRACT(_normalize(a["href"])).registered_domain.lower()
        if not href_dom:
            continue
        anchor = a.get_text(" ").strip()
        if "." not in anchor:
            continue
        # Pull the domain-looking token out of the anchor text.
        for tok in anchor.replace("/", " ").split():
            if "." in tok:
                anchor_dom = _EXTRACT(_normalize(tok)).registered_domain.lower()
                if anchor_dom and anchor_dom != href_dom:
                    return True
    return False
