"""Targeted augmentation: legitimate transactional / service-notification emails.

WHY THIS EXISTS
---------------
The real corpus (CEAS/Nazario/Enron/...) has a blind spot: its *legitimate*
class is business/personal/mailing-list email and contains almost no modern
SERVICE NOTIFICATIONS (order receipts, shipping updates, statements, account
welcomes, sign-in notices). Meanwhile phishing is full of "your account",
"sign in", "log in" phrasing. Measured on the corpus, "your account" is ~91%
phishing. So both the baseline and DistilBERT wrongly flag a *legitimate*
"your statement is ready, log in to your account" email as phishing.

This is a DATA COVERAGE gap, not a model-capacity gap — a bigger model makes
it worse. The fix is to add legitimate examples that use this exact vocabulary
*without* phishing cues so the model learns the phrasing is not inherently
malicious.

This is a transparent, OPTIONAL augmentation layered on top of the ~82k real
emails (not a stand-alone training set). It is generated from many templates
× many brands × randomized details, yielding thousands of distinct emails —
not a handful of near-duplicates. Every sample is built with strict
anti-phishing rules:
  * real, correctly-spelled brand domains (no lookalikes)
  * NO urgency / deadlines / threats
  * NO request to "verify your identity" or enter a password
  * links (when present) point to the real brand domain

Run standalone to preview:
    python src/generate_legit_transactional.py --preview 8
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import clean_text # noqa: E402

# (brand, primary domain) — real, correctly spelled.
BRANDS = [
    ("Amazon", "amazon.com"), ("PayPal", "paypal.com"), ("Netflix", "netflix.com"),
    ("Spotify", "spotify.com"), ("Google", "accounts.google.com"),
    ("Microsoft", "microsoft.com"), ("Apple", "apple.com"), ("Dropbox", "dropbox.com"),
    ("LinkedIn", "linkedin.com"), ("GitHub", "github.com"), ("Slack", "slack.com"),
    ("Uber", "uber.com"), ("Airbnb", "airbnb.com"), ("eBay", "ebay.com"),
    ("Shopify", "shopify.com"), ("Zoom", "zoom.us"), ("Adobe", "adobe.com"),
    ("Steam", "steampowered.com"), ("Notion", "notion.so"), ("Figma", "figma.com"),
    ("First National Bank", "firstnational.com"), ("Stripe", "stripe.com"),
    ("Etsy", "etsy.com"), ("Booking.com", "booking.com"), ("Coursera", "coursera.org"),
]
# Real bank / card-issuer brands — used by the bank-statement builders below.
# These cover a documented residual blind spot: legitimate "bank statement /
# online banking / credit card statement" notifications, whose vocabulary is
# ~91% phishing in the raw corpus, so Stage 1 wrongly flags them.
BANK_BRANDS = [
    ("Chase", "chase.com"), ("Wells Fargo", "wellsfargo.com"),
    ("Bank of America", "bankofamerica.com"), ("HSBC", "hsbc.com"),
    ("American Express", "americanexpress.com"), ("Citi", "citi.com"),
    ("Capital One", "capitalone.com"), ("U.S. Bank", "usbank.com"),
    ("PNC Bank", "pnc.com"), ("TD Bank", "td.com"), ("Barclays", "barclays.co.uk"),
    ("First National Bank", "firstnational.com"),
]
# Fraction of generated legit emails that use the bank-statement builders.
BANK_FRACTION = 0.35

FIRST_NAMES = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Chris", "Jamie",
               "Riley", "Casey", "Drew", "Sarah", "David", "Maria", "Liam",
               "Priya", "Noah", "Emma", "Lucas", "Olivia", "Ethan"]
ITEMS = ["wireless headphones", "running shoes", "a coffee maker", "a paperback novel",
         "an HDMI cable", "a desk lamp", "yoga mat", "a phone case", "a backpack",
         "kitchen knives", "a board game", "an external SSD", "a water bottle",
         "winter gloves", "a mechanical keyboard"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def _order_id(rng: random.Random) -> str:
    return f"{rng.randint(100,999)}-{rng.randint(1000000,9999999)}-{rng.randint(1000000,9999999)}"


def _amount(rng: random.Random) -> str:
    return f"${rng.randint(5,480)}.{rng.randint(0,99):02d}"


def _date(rng: random.Random) -> str:
    return f"{rng.choice(MONTHS)} {rng.randint(1,28)}, {rng.choice([2024,2025,2026])}"


# Each builder returns (subject, body). They deliberately use phrasing that
# currently triggers false positives ("log in to your account", "sign in",
# "your account", "statement") but in a benign, no-urgency context.
def _order_confirmation(b, dom, name, rng):
    item = rng.choice(ITEMS)
    return (f"Your {b} order has shipped",
            f"Hi {name}, thanks for your order. Your {item} (order {_order_id(rng)}) "
            f"has shipped and is on its way. The total was {_amount(rng)}. "
            f"You can track the delivery any time by signing in to your account at https://{dom}. "
            f"Thanks for shopping with {b}.")

def _receipt(b, dom, name, rng):
    return (f"Your {b} receipt for {_date(rng)}",
            f"Hello {name}, this is a receipt for your recent purchase of {_amount(rng)}. "
            f"No action is needed. A copy is saved in your account history at https://{dom}. "
            f"Thank you for being a {b} customer.")

def _statement(b, dom, name, rng):
    return (f"Your {b} statement is ready",
            f"Hi {name}, your monthly statement for {rng.choice(MONTHS)} is now available. "
            f"You can log in to your account at https://{dom} to view your transactions and balance. "
            f"There is nothing you need to do.")

def _subscription(b, dom, name, rng):
    return (f"Your {b} subscription has renewed",
            f"Hi {name}, your {b} subscription renewed for {_amount(rng)} on {_date(rng)}. "
            f"Manage your plan or payment method any time from your account at https://{dom}.")

def _welcome(b, dom, name, rng):
    return (f"Welcome to {b}, {name}",
            f"Welcome aboard, {name}! Your {b} account is all set up. "
            f"Sign in at https://{dom} to explore your dashboard and personalize your settings. "
            f"We're glad to have you.")

def _shipping(b, dom, name, rng):
    return (f"{b}: your package is out for delivery",
            f"Good news {name} — your package is out for delivery and should arrive today. "
            f"Track its progress from your account at https://{dom}. Tracking number "
            f"{rng.randint(10**11, 10**12-1)}.")

def _newsletter(b, dom, name, rng):
    topic = rng.choice(["product updates", "this week's highlights", "tips and tricks",
                        "new features", "community picks"])
    return (f"{b} Digest: {topic}",
            f"Hi {name}, here's your {b} newsletter with {topic}. "
            f"Read the full digest on our site at https://{dom}. "
            f"You are receiving this because you subscribed; you can update your preferences any time.")

def _meeting(b, dom, name, rng):
    return (f"Invitation: team sync on {_date(rng)}",
            f"Hi {name}, you've been invited to a meeting. The agenda and video link "
            f"are attached to the calendar event. Add it to your calendar from {b} at https://{dom}.")

def _file_share(b, dom, name, rng):
    other = rng.choice(FIRST_NAMES)
    return (f"{other} shared a document with you on {b}",
            f"Hi {name}, {other} shared a file with you. "
            f"You can open it from your shared folder by signing in to {b} at https://{dom}.")

def _signin_notice(b, dom, name, rng):
    city = rng.choice(["Paris", "Berlin", "Chicago", "Toronto", "Madrid", "Lyon"])
    return (f"New sign-in to your {b} account",
            f"Hi {name}, we noticed a new sign-in to your account from {city}. "
            f"If this was you, no action is needed. You can review your active sessions "
            f"any time in your account settings at https://{dom}.")

def _password_changed(b, dom, name, rng):
    return (f"Your {b} password was changed",
            f"Hi {name}, this is a confirmation that your password was changed on {_date(rng)}. "
            f"If you made this change, you're all set and can sign in as usual at https://{dom}.")

BUILDERS = [
    _order_confirmation, _receipt, _statement, _subscription, _welcome,
    _shipping, _newsletter, _meeting, _file_share, _signin_notice, _password_changed,
]


# Bank-statement builders. They deliberately use the FP-triggering vocabulary
# ("bank statement", "online banking", "credit card statement", "your account",
# "balance") in a benign, no-urgency, no-credential-request context.
def _bank_statement(b, dom, name, rng):
    return (f"Your {b} statement is ready",
            f"Hi {name}, your monthly bank statement for {rng.choice(MONTHS)} is now "
            f"available to view in online banking. You can sign in to your account at "
            f"https://{dom} to download a copy. There is nothing you need to do.")

def _cc_statement(b, dom, name, rng):
    return (f"Your {b} credit card statement is available",
            f"Hello {name}, your credit card statement for {rng.choice(MONTHS)} is ready. "
            f"Your statement balance is {_amount(rng)}, due on {_date(rng)}. "
            f"You can view it any time in online banking at https://{dom}.")

def _online_banking_notice(b, dom, name, rng):
    return (f"{b}: your monthly account summary",
            f"Hi {name}, here is your monthly account summary. Your current balance is "
            f"{_amount(rng)}. You can review your recent transactions in online banking "
            f"at https://{dom}. No action is required.")

def _deposit_notice(b, dom, name, rng):
    return (f"{b}: a deposit has posted to your account",
            f"Hi {name}, a deposit of {_amount(rng)} has posted to your account. "
            f"You can see the details in online banking by signing in at https://{dom}.")

def _payment_received(b, dom, name, rng):
    return (f"{b}: we received your payment",
            f"Hi {name}, thank you — we received your payment of {_amount(rng)} on "
            f"{_date(rng)}. Your next statement will be available {_date(rng)}. "
            f"Manage your account any time at https://{dom}.")

BANK_BUILDERS = [
    _bank_statement, _cc_statement, _online_banking_notice,
    _deposit_notice, _payment_received,
]


def generate_legit_df(n: int = 2800, seed: int = 42) -> pd.DataFrame:
    """Generate ~n distinct legitimate transactional emails (label = 0).

    A BANK_FRACTION share uses the bank-statement builders (real bank domains)
    to cover the legit financial-notification blind spot."""
    rng = random.Random(seed)
    seen: set[str] = set()
    rows: list[str] = []
    attempts = 0
    while len(rows) < n and attempts < n * 30:
        attempts += 1
        if rng.random() < BANK_FRACTION:
            build = rng.choice(BANK_BUILDERS)
            brand, dom = rng.choice(BANK_BRANDS)
        else:
            build = rng.choice(BUILDERS)
            brand, dom = rng.choice(BRANDS)
        name = rng.choice(FIRST_NAMES)
        subject, body = build(brand, dom, name, rng)
        text = clean_text(f"{subject} {body}")
        if text in seen:
            continue
        seen.add(text)
        rows.append(text)
    return pd.DataFrame({"text": rows, "label": 0})


def main() -> int:
    ap = argparse.ArgumentParser(description="Aperçu de l'augmentation légitime")
    ap.add_argument("--preview", type=int, default=6)
    ap.add_argument("-n", type=int, default=2000)
    args = ap.parse_args()
    df = generate_legit_df(n=args.n)
    print(f"{len(df)} emails légitimes transactionnels distincts générés.\n")
    for t in df["text"].head(args.preview):
        print("-", t[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
