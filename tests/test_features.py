"""Phase 1 verification — URL + text feature extractors.

Run: python tests/test_features.py   (plain asserts, no pytest needed)

Success criteria (from the v2 plan):
- No legitimate URL (amazon.fr, github.com) scores > 0.3 on TLD risk.
- 5 sanity examples: 2 phishing, 2 legitimate, 1 ambiguous behave sensibly.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features.url_features import extract_url_features  # noqa: E402
from features.text_features import extract_text_features  # noqa: E402


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  [OK] {msg}")


def test_legit_urls_low_tld_risk():
    print("\n# legit URLs must not score > 0.3 TLD risk")
    for url in ("https://www.amazon.fr/order/123", "https://github.com/x/y",
                "https://zoom.us/j/123", "https://stripe.com/pay/inv_1"):
        f = extract_url_features(url)
        check(f["max_tld_risk_score"] <= 0.3, f"{url} -> tld_risk={f['max_tld_risk_score']}")
        check(f["domain_typosquatting_score"] == 0.0, f"{url} -> no typosquat")


def test_phishing_url_signals():
    print("\n# phishing URLs must light up structural features")
    f = extract_url_features("http://amaz0n-secure.tk/login?redirect=verify")
    check(f["max_tld_risk_score"] == 1.0, "amaz0n-secure.tk -> tld_risk 1.0")
    check(f["has_auth_keyword_in_path"], "login in path detected")
    check(f["has_redirect_keyword"], "redirect in query detected")
    check(f["domain_typosquatting_score"] >= 0.7, f"amaz0n typosquat={f['domain_typosquatting_score']}")

    f2 = extract_url_features("http://192.168.0.5/account/verify")
    check(f2["has_ip_url"], "raw IP host detected")


def test_no_url():
    print("\n# text with no URL")
    f = extract_url_features("Hi team, see you at the meeting tomorrow.")
    check(not f["has_url"] and f["url_count"] == 0, "no url flags clear")
    check(f["legitimate_domain_ratio"] == 0.0, "legit ratio 0 when no url")


def test_financial_context_disambiguation():
    print("\n# invoice alone != bank_alert; combination matters")
    invoice = extract_text_features(
        "Please find attached your invoice for this month. Receipt #123.")
    check(invoice["financial_context"] == "invoice", "plain invoice -> 'invoice'")
    check(not invoice["credential_request"], "invoice has no credential request")

    bank = extract_text_features(
        "Your account has been blocked due to unusual activity. "
        "Verify your identity and enter your password now.")
    check(bank["financial_context"] == "bank_alert", "bank alert -> 'bank_alert'")
    check(bank["credential_request"], "bank alert asks for credentials")


def test_ambiguous_example():
    print("\n# ambiguous: urgency + delivery, legit-looking domain")
    text = ("Your parcel could not be delivered. Track it here: "
            "https://www.chronopost.fr/suivi/123 — please confirm by tomorrow.")
    uf = extract_url_features(text)
    tf = extract_text_features(text)
    check(uf["max_tld_risk_score"] == 0.0, "chronopost.fr -> no tld risk")
    check(uf["legitimate_domain_ratio"] == 1.0, "chronopost is a known brand domain")
    check(tf["package_delivery_context"], "delivery context detected")
    # No verdict here — Stage 2 only measures.


def main() -> int:
    test_legit_urls_low_tld_risk()
    test_phishing_url_signals()
    test_no_url()
    test_financial_context_disambiguation()
    test_ambiguous_example()
    print("\nAll Phase 1 feature checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
