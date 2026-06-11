"""Targeted augmentation: modern phishing emails with realistic malicious URLs.

WHY THIS EXISTS
---------------
The real corpus (CEAS/Nazario/Enron/...) has a symmetric blind spot to the one
fixed by generate_legit_transactional.py: its *phishing* class is old-spam /
Nazario-era lures whose URLs look ordinary. Measured on the corpus, the modern
malicious-URL signals are essentially ABSENT from phishing:
  * suspicious TLD (.tk/.ml/.xyz/...) present in ~0% of phishing URLs
  * typosquatting / lookalike domains: ~0.2% of phishing URLs
  * raw-IP hosts: ~1%
So a URL submodel trained on the corpus learns `url_count` (a spam-volume proxy)
instead of the real phishing structure, and scores a textbook
`amaz0n-secure.tk/login?redirect=verify` URL as only ~0.5.

This is a DATA COVERAGE gap, not a model-capacity gap — the exact same situation
(and fix) as the legitimate-transactional augmentation. We add transparent,
rule-generated phishing emails that DO exhibit modern URL tricks (lookalike
domains, suspicious TLDs, IP hosts, auth keywords + redirects in the path),
combined with the textual lures phishing actually uses (urgency, account
threats, credential requests). Many templates × many impersonated brands ×
randomized lookalike generators × randomized details → thousands of distinct
emails, not near-duplicates.

This is OPTIONAL augmentation layered on the ~82k real emails. Set the knob
SYNTHETIC_PHISH_N in download_dataset.py to 0 to disable.

Run standalone to preview:
    python src/generate_phishing_url.py --preview 8
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import clean_text # noqa: E402

# Brands phishers impersonate (the label they put in the email). The generated
# URLs are LOOKALIKES of these — never the real domain.
TARGETS = [
    "Amazon", "PayPal", "Apple", "Microsoft", "Netflix", "Google", "LinkedIn",
    "DHL", "FedEx", "UPS", "Chronopost", "La Poste", "DocuSign", "Dropbox",
    "Bank of America", "Wells Fargo", "Chase", "HSBC", "Coinbase", "Outlook",
    "Instagram", "Facebook", "WhatsApp", "Orange", "Booking.com",
]

# Suspicious / frequently-abused TLDs (match url_features.SUSPICIOUS_TLDS).
SUS_TLDS = ["tk", "ml", "ga", "cf", "gq", "xyz", "top", "click",
            "download", "work", "party"]
AUTH_PATHS = ["login", "verify", "secure/account", "account/verify",
              "update-password", "signin", "confirm-identity",
              "webscr/login", "auth/verify", "account/recovery"]
REDIRECTS = ["?redirect=verify", "?url=login", "?next=secure", "&link=account",
             "?return=account/verify", "?goto=signin", ""]
DECOR = ["secure", "verify", "account", "support", "login", "service", "alert"]

FIRST_NAMES = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Chris", "Jamie",
               "Riley", "Casey", "Drew", "Sarah", "David", "Maria", "Liam",
               "Priya", "Noah", "Emma", "Lucas", "Olivia", "Ethan"]
CITIES = ["Lagos", "Moscow", "Kiev", "Beijing", "an unknown location",
          "Jakarta", "Bucharest", "Cairo", "Hanoi"]


def _brand_slug(brand: str) -> str:
    return brand.lower().replace(" ", "").replace(".", "")


def _typosquat(slug: str, rng: random.Random) -> str:
    """Return a misspelled / decorated lookalike of a brand slug."""
    s = slug
    swaps = {"o": "0", "l": "1", "i": "1", "a": "4", "e": "3", "s": "5"}
    # apply 1-2 character swaps
    for _ in range(rng.randint(1, 2)):
        for ch, repl in swaps.items():
            if ch in s and rng.random() < 0.5:
                s = s.replace(ch, repl, 1)
                break
    kind = rng.random()
    if kind < 0.4:
        return f"{s}-{rng.choice(DECOR)}"
    if kind < 0.7:
        return f"{rng.choice(DECOR)}-{s}"
    return s


def _ip(rng: random.Random) -> str:
    return ".".join(str(rng.randint(1, 254)) for _ in range(4))


def _malicious_url(brand: str, rng: random.Random) -> str:
    """Generate a varied malicious-looking URL for an impersonated brand."""
    slug = _brand_slug(brand)
    path = rng.choice(AUTH_PATHS)
    redirect = rng.choice(REDIRECTS)
    kind = rng.random()
    if kind < 0.35: # typosquat + suspicious TLD
        host = f"{_typosquat(slug, rng)}.{rng.choice(SUS_TLDS)}"
    elif kind < 0.55: # brand-name + suspicious TLD
        host = f"{slug}.{rng.choice(SUS_TLDS)}"
    elif kind < 0.75: # stacked subdomains
        host = f"{rng.choice(DECOR)}.{rng.choice(DECOR)}.{slug}.{rng.choice(SUS_TLDS)}"
    elif kind < 0.88: # raw IP host
        host = _ip(rng)
    else: # brand-as-subdomain of junk
        host = f"{slug}-{rng.choice(DECOR)}.{rng.choice(['com','net','info'])}"
    scheme = "http" if rng.random() < 0.6 else "https"
    return f"{scheme}://{host}/{path}{redirect}"


def _deadline(rng: random.Random) -> str:
    return rng.choice(["24 hours", "48 hours", "today", "the next 12 hours",
                       "2 hours", "the end of the day"])


def _amount(rng: random.Random) -> str:
    return f"{rng.choice(['$','€','£'])}{rng.randint(1,9)}.{rng.randint(0,99):02d}"


# Each builder returns (subject, body) and embeds one malicious URL. They use
# the lures real phishing uses: urgency, account threats, credential requests,
# impersonal greetings.
def _account_suspended(b, url, name, rng):
    return (f"[Action Required] Your {b} account has been suspended",
            f"Dear Customer, we detected unusual activity and your {b} account has been "
            f"temporarily blocked. You must verify your identity within {_deadline(rng)} "
            f"or your account will be permanently suspended. Confirm your password here: {url}")

def _bank_alert(b, url, name, rng):
    return (f"{b} Security Alert: unauthorized sign-in",
            f"Dear user, a new sign-in to your account was detected from {rng.choice(CITIES)}. "
            f"If this wasn't you, your account is at risk. Verify now to secure it: {url} "
            f"Failure to act within {_deadline(rng)} will result in account closure.")

def _package_fee(b, url, name, rng):
    return (f"{b}: your parcel is on hold",
            f"Dear customer, your package could not be delivered because a customs fee of "
            f"{_amount(rng)} is unpaid. Please confirm your details and pay now to release "
            f"your parcel, otherwise it will be returned: {url}")

def _password_expiry(b, url, name, rng):
    return (f"Your {b} password expires {_deadline(rng)}",
            f"Dear user, for security reasons your password will expire soon. To keep access "
            f"to your account, you must log in and update your credentials immediately: {url}")

def _payment_confirm(b, url, name, rng):
    return (f"{b}: confirm your recent payment",
            f"Dear Customer, we were unable to validate your last payment of {_amount(rng)}. "
            f"Your account will be limited unless you re-enter your card details and confirm "
            f"your identity within {_deadline(rng)}: {url}")

def _refund_bait(b, url, name, rng):
    return (f"{b} refund of {_amount(rng)} pending",
            f"Dear user, you are eligible for a refund of {_amount(rng)}. To receive it you "
            f"must verify your account and confirm your banking information here: {url} "
            f"This offer expires in {_deadline(rng)}.")

def _doc_share(b, url, name, rng):
    return (f"You have a secure {b} document to sign",
            f"Dear Customer, a confidential document is waiting for your signature. "
            f"Sign in to verify your identity and access it before it expires {_deadline(rng)}: {url}")

BUILDERS = [
    _account_suspended, _bank_alert, _package_fee, _password_expiry,
    _payment_confirm, _refund_bait, _doc_share,
]


def generate_phishing_df(n: int = 2500, seed: int = 7) -> pd.DataFrame:
    """Generate ~n distinct phishing emails with modern malicious URLs (label=1)."""
    rng = random.Random(seed)
    seen: set[str] = set()
    rows: list[str] = []
    attempts = 0
    while len(rows) < n and attempts < n * 40:
        attempts += 1
        build = rng.choice(BUILDERS)
        brand = rng.choice(TARGETS)
        name = rng.choice(FIRST_NAMES)
        url = _malicious_url(brand, rng)
        subject, body = build(brand, url, name, rng)
        text = clean_text(f"{subject} {body}")
        if text in seen:
            continue
        seen.add(text)
        rows.append(text)
    return pd.DataFrame({"text": rows, "label": 1})


def main() -> int:
    ap = argparse.ArgumentParser(description="Aperçu de l'augmentation phishing")
    ap.add_argument("--preview", type=int, default=6)
    ap.add_argument("-n", type=int, default=2500)
    args = ap.parse_args()
    df = generate_phishing_df(n=args.n)
    print(f"{len(df)} emails de phishing distincts générés.\n")
    for t in df["text"].head(args.preview):
        print("-", t[:220])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
