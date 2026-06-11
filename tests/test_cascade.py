"""Phase 5 verification — cascade non-regression gate.

Run: python tests/test_cascade.py

Binding success gate (chosen for v2):
  * No MUST_BE_LEGITIMATE email exceeds risk_score 45.
  * No MUST_BE_PHISHING email drops below risk_score 70.
  * Explanations never cite a single signal ("a link is present") on its own.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import predict_email  # noqa: E402

MUST_BE_LEGITIMATE = [
    "Hi John, please find attached your invoice for this month. Payment link: https://stripe.com/pay/inv_123",
    "Your Amazon order #123-456 has shipped. Track it at: https://www.amazon.fr/order/123-456",
    "Meeting at 3pm tomorrow, here's the Zoom link: https://zoom.us/j/123456789",
    "Please reset your GitHub password: https://github.com/password_reset/token_abc",
]

MUST_BE_PHISHING = [
    "URGENT: Your bank account will be suspended. Verify now: http://secure-bank.tk/verify?id=1",
    "Dear Customer, click to confirm your PayPal password: http://paypa1-secure.ml/login",
    "You have a package. Pay 2 EUR customs fee: http://chronopost-fr.xyz/pay",
]

# Broader set of legitimate emails that contain links — none should exceed 40.
LEGIT_WITH_LINKS = [
    "Your monthly statement is ready. Sign in to view it: https://www.paypal.com/myaccount",
    "Thanks for your purchase! Your receipt is at https://www.apple.com/receipt/9981",
    "Your flight itinerary is confirmed. Manage your booking: https://www.booking.com/trip/55",
    "Here is the document you requested: https://docs.google.com/document/d/abc/edit",
    "Your Netflix subscription renewed. Manage your plan at https://www.netflix.com/account",
    "New comment on your pull request: https://github.com/acme/repo/pull/42",
    "Your Spotify Wrapped is here! See it at https://www.spotify.com/wrapped",
    "Invoice #884 attached. Pay securely at https://dashboard.stripe.com/invoices/884",
    "Your order has shipped via UPS, track at https://www.ups.com/track?id=1Z999",
    "Slack: you were mentioned in #general. Open: https://acme.slack.com/archives/C01",
    "Your Dropbox file was shared. View it: https://www.dropbox.com/s/abc/report.pdf",
    "Reminder: team standup at 9am. Join: https://zoom.us/j/9876543210",
    "Your Adobe license is active. Manage it at https://account.adobe.com/plans",
    "Your LinkedIn weekly digest: https://www.linkedin.com/feed/news",
    "Your boarding pass is ready: https://www.airfrance.fr/boarding/AF1234",
    "Your monthly usage report from AWS: https://console.aws.amazon.com/billing",
    "Microsoft 365: your subscription receipt https://account.microsoft.com/billing",
    "Your Coursera certificate is ready: https://www.coursera.org/account/accomplishments",
    "Your eBay item sold! Print the label at https://www.ebay.com/sh/ord/labels",
    "Your bank statement for May is available in online banking at https://www.chase.com/personal",
]


# French emails — the model must handle French, not flag it as phishing by default.
FR_LEGIT = [
    "Bonjour Marie, je te confirme notre réunion de demain à 14h en salle 3. À demain, Paul.",
    "Madame, Monsieur, votre facture du mois de mai est disponible. Montant 49,99 EUR. Consultez-la dans votre espace client sur https://www.free.fr/account.",
    "Bonjour, nous vous confirmons votre commande numéro 884512. Expédiée sous 48h. Suivez votre colis sur https://www.laposte.fr/suivi.",
    "Salut, on se voit ce week-end ? Tiens-moi au courant. Bises.",
    "Bonjour, votre relevé de compte du mois d'avril est disponible dans votre banque en ligne sur https://www.labanquepostale.fr.",
    "Rappel : votre rendez-vous est prévu le 12 juin à 9h. Merci de prévenir en cas d'empêchement.",
]
FR_PHISH = [
    "Cher client, votre compte a été suspendu. Vérifiez votre identité maintenant : http://laposte-secure.tk/connexion",
    "URGENT : nouvelle connexion à votre compte bancaire. Confirmez votre mot de passe : http://bnp-verif.ml/securite",
    "Remboursement d'impôts de 240 EUR en attente. Confirmez vos coordonnées bancaires sous 24h : http://impots-gouv.xyz/remboursement",
]


def run(label, emails, *, max_legit=None, min_phish=None):
    print(f"\n# {label}")
    failures = []
    for text in emails:
        r = predict_email(text)
        score = r["risk_score"]
        ok = True
        if max_legit is not None and score > max_legit:
            ok = False
        if min_phish is not None and score < min_phish:
            ok = False
        # explanations must never be a lone bare-link / bare-payment citation
        for e in r["explanations"]:
            el = e.lower()
            if el.strip() in ("présence de lien", "lien détecté", "paiement mentionné"):
                ok = False
        mark = "OK " if ok else "FAIL"
        if not ok:
            failures.append((score, text[:60], r["verdict"], r["explanations"]))
        print(f"  [{mark}] {score:5.1f}  {r['verdict']:<10} {text[:60]}")
    return failures


def main() -> int:
    fails = []
    fails += run("MUST_BE_LEGITIMATE (<= 45)", MUST_BE_LEGITIMATE, max_legit=45)
    fails += run("MUST_BE_PHISHING (>= 70)", MUST_BE_PHISHING, min_phish=70)
    fails += run("LEGIT_WITH_LINKS (<= 40)", LEGIT_WITH_LINKS, max_legit=40)
    fails += run("FR_LEGIT (<= 45)", FR_LEGIT, max_legit=45)
    fails += run("FR_PHISH (>= 70)", FR_PHISH, min_phish=70)

    print("\n" + "=" * 60)
    if fails:
        print(f"FAILED: {len(fails)} case(s) out of bounds:")
        for score, snip, verdict, expl in fails:
            print(f"  {score:5.1f} {verdict:<10} {snip}")
            for e in expl:
                print(f"        - {e}")
        return 1
    print("All cascade non-regression checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
