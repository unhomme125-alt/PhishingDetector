"""Targeted augmentation: FRENCH emails (both classes).

WHY THIS EXISTS
---------------
The real corpus (CEAS/Nazario/Enron/...) and the prior augmentations are ~100%
English. French legitimate email is therefore out-of-distribution and the model
flags it as phishing by default (a benign French "réunion demain 14h" scores
~99%, vs ~1% for the same sentence in English). This is a LANGUAGE coverage gap.

We add transparent, rule-generated FRENCH emails in BOTH classes so the model
learns French vocabulary *discriminatively* — not "French = legitimate" (which
would miss French phishing). Many templates × brands/details → thousands of
distinct emails. Legitimate emails follow strict anti-phishing rules (real
French brand domains, no urgency, no credential requests, and some with NO link
at all so short personal/admin mails aren't auto-flagged). Phishing emails use
the lures French phishing actually uses (urgence, menace de suspension, demande
d'identifiants, URL malveillante / typosquat / TLD suspect).

Run standalone to preview:
    python src/generate_french.py --preview 6
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import clean_text # noqa: E402

PRENOMS = ["Marie", "Paul", "Julie", "Thomas", "Camille", "Lucas", "Léa",
           "Hugo", "Chloé", "Nicolas", "Manon", "Antoine", "Sarah", "Maxime",
           "Emma", "Théo", "Inès", "Louis", "Jade", "Gabriel"]
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
        "août", "septembre", "octobre", "novembre", "décembre"]
ARTICLES = ["un casque audio", "une paire de chaussures", "un livre", "un câble HDMI",
            "une lampe de bureau", "un sac à dos", "un clavier", "une coque de téléphone",
            "un jeu de société", "une bouilloire", "des écouteurs", "un disque dur externe"]

# Real French (or FR-facing) brand domains — for the LEGITIMATE class.
MARQUES = [
    ("Amazon", "amazon.fr"), ("La Poste", "laposte.fr"), ("Free", "free.fr"),
    ("Orange", "orange.fr"), ("SFR", "sfr.fr"), ("Fnac", "fnac.com"),
    ("Cdiscount", "cdiscount.com"), ("SNCF Connect", "sncf-connect.com"),
    ("Doctolib", "doctolib.fr"), ("Ameli", "ameli.fr"), ("Chronopost", "chronopost.fr"),
    ("La Banque Postale", "labanquepostale.fr"), ("Crédit Mutuel", "creditmutuel.fr"),
    ("BNP Paribas", "mabanque.bnpparibas"), ("Boulanger", "boulanger.com"),
    ("Leroy Merlin", "leroymerlin.fr"), ("Deezer", "deezer.com"),
]

# Brands impersonated in the PHISHING class (lookalike URLs only).
CIBLES = ["Amazon", "La Poste", "Chronopost", "Ameli", "Impots", "Free", "Orange",
          "PayPal", "Netflix", "La Banque Postale", "Crédit Agricole", "BNP",
          "SFR", "Colissimo", "Doctolib"]
SUS_TLDS = ["tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "work"]
CHEMINS = ["connexion", "verifier", "securite/compte", "compte/verification",
           "mot-de-passe", "identification", "paiement", "remboursement"]
DECOR = ["securite", "verif", "compte", "support", "service", "alerte", "client"]
DELAIS = ["24 heures", "48 heures", "aujourd'hui", "les 12 prochaines heures",
          "2 heures", "la fin de la journée"]
VILLES = ["Lagos", "Moscou", "Kiev", "Pékin", "un lieu inconnu", "Bucarest"]


def _id(rng):
    return f"{rng.randint(100,999)}-{rng.randint(1000000,9999999)}"


def _montant(rng):
    return f"{rng.randint(2,480)},{rng.randint(0,99):02d} €"


def _date(rng):
    return f"{rng.randint(1,28)} {rng.choice(MOIS)} {rng.choice([2024,2025,2026])}"


# ---------------------------------------------------------------- LÉGITIME ----
def _l_reunion(b, dom, name, rng):
    return ("Réunion de demain",
            f"Bonjour {name}, je te confirme notre réunion de demain à "
            f"{rng.randint(9,17)}h en salle {rng.randint(1,8)}. N'hésite pas si tu as "
            f"des points à ajouter à l'ordre du jour. À demain.")

def _l_commande(b, dom, name, rng):
    return (f"Confirmation de votre commande {b}",
            f"Bonjour {name}, nous vous confirmons la réception de votre commande "
            f"n°{_id(rng)} ({rng.choice(ARTICLES)}). Le montant de {_montant(rng)} a "
            f"bien été pris en compte. Vous pouvez suivre votre commande depuis votre "
            f"espace client sur https://{dom}. Merci de votre confiance.")

def _l_facture(b, dom, name, rng):
    return (f"Votre facture {b} est disponible",
            f"Bonjour {name}, votre facture du mois de {rng.choice(MOIS)} d'un montant "
            f"de {_montant(rng)} est disponible dans votre espace client sur https://{dom}. "
            f"Aucune action n'est requise de votre part.")

def _l_releve(b, dom, name, rng):
    return (f"Votre relevé {b} est disponible",
            f"Bonjour {name}, votre relevé de compte du mois de {rng.choice(MOIS)} est "
            f"désormais consultable dans votre banque en ligne. Connectez-vous à votre "
            f"espace sur https://{dom} pour le consulter. Il n'y a rien à faire.")

def _l_livraison(b, dom, name, rng):
    return (f"{b} : votre colis est en cours de livraison",
            f"Bonjour {name}, votre colis est en cours de livraison et devrait arriver "
            f"aujourd'hui. Vous pouvez suivre son acheminement depuis votre espace sur "
            f"https://{dom}. Numéro de suivi : {rng.randint(10**9,10**10)}.")

def _l_rdv(b, dom, name, rng):
    return ("Rappel de votre rendez-vous",
            f"Bonjour {name}, nous vous rappelons votre rendez-vous le {_date(rng)}. "
            f"En cas d'empêchement, merci de prévenir à l'avance. Bonne journée.")

def _l_perso(b, dom, name, rng):
    autre = rng.choice(PRENOMS)
    return ("Des nouvelles",
            f"Salut {name}, j'espère que tu vas bien ! Je voulais juste te dire que "
            f"{autre} passe ce week-end, on pourrait se voir si tu es dispo. "
            f"Tiens-moi au courant. Bises.")

def _l_rh(b, dom, name, rng):
    return ("Information RH : congés",
            f"Bonjour {name}, pour rappel, merci de poser vos congés d'été avant le "
            f"{_date(rng)} via l'intranet. N'hésitez pas à revenir vers le service RH "
            f"pour toute question. Cordialement.")

def _l_newsletter(b, dom, name, rng):
    return (f"La lettre {b} de la semaine",
            f"Bonjour {name}, découvrez nos actualités et nos sélections de la semaine. "
            f"Pour lire l'intégralité, rendez-vous sur https://{dom}. Vous recevez cet "
            f"email car vous êtes inscrit ; vous pouvez vous désabonner à tout moment.")

def _l_recu(b, dom, name, rng):
    return (f"Votre reçu {b}",
            f"Bonjour {name}, voici le reçu de votre paiement de {_montant(rng)} du "
            f"{_date(rng)}. Aucune action n'est nécessaire. Une copie est conservée dans "
            f"votre espace client. Merci.")

# Short, anchorless transactional notices (no brand, link, name or id) — covers
# generic one-liners like "merci pour votre commande, expédiée sous 48h" that
# otherwise look out-of-distribution.
def _l_commande_courte(b, dom, name, rng):
    return ("Merci pour votre commande",
            f"Bonjour, merci pour votre commande. Elle sera expédiée sous "
            f"{rng.choice(['24','48','72'])} heures. Vous recevrez un email de suivi dès "
            f"l'expédition. Bonne journée.")

def _l_expedie_courte(b, dom, name, rng):
    return ("Votre commande a été expédiée",
            f"Bonjour, votre commande vient d'être expédiée et arrivera sous "
            f"{rng.randint(2,5)} jours ouvrés. Merci pour votre achat.")

def _l_paiement_courte(b, dom, name, rng):
    return ("Paiement bien reçu",
            f"Bonjour, nous avons bien reçu votre paiement de {_montant(rng)}. "
            f"Aucune action n'est nécessaire de votre part. Merci de votre confiance.")

# Mix of builders with a link (transactional) and without (short personal/admin:
# _l_reunion, _l_rdv, _l_perso, _l_rh) so anchorless legit mail isn't auto-flagged.
LEGIT_BUILDERS = [
    _l_reunion, _l_commande, _l_facture, _l_releve, _l_livraison, _l_rdv,
    _l_perso, _l_rh, _l_newsletter, _l_recu,
    _l_commande_courte, _l_expedie_courte, _l_paiement_courte,
]


# ---------------------------------------------------------------- PHISHING ----
def _slug(b):
    return (b.lower().replace(" ", "").replace("é", "e").replace("è", "e")
            .replace("ô", "o").replace("'", ""))


def _typo(slug, rng):
    s = slug
    for ch, repl in {"o": "0", "l": "1", "i": "1", "a": "4", "e": "3"}.items():
        if ch in s and rng.random() < 0.5:
            s = s.replace(ch, repl, 1)
            break
    return f"{s}-{rng.choice(DECOR)}" if rng.random() < 0.6 else f"{rng.choice(DECOR)}-{s}"


def _url_mal(b, rng):
    slug = _slug(b)
    chemin = rng.choice(CHEMINS)
    k = rng.random()
    if k < 0.4:
        host = f"{_typo(slug, rng)}.{rng.choice(SUS_TLDS)}"
    elif k < 0.6:
        host = f"{slug}.{rng.choice(SUS_TLDS)}"
    elif k < 0.8:
        host = f"{rng.choice(DECOR)}.{slug}.{rng.choice(SUS_TLDS)}"
    else:
        host = ".".join(str(rng.randint(1, 254)) for _ in range(4)) # IP
    scheme = "http" if rng.random() < 0.7 else "https"
    return f"{scheme}://{host}/{chemin}"


def _p_compte(b, url, name, rng):
    return (f"[Action requise] Votre compte {b} a été suspendu",
            f"Cher client, nous avons détecté une activité inhabituelle et votre compte "
            f"{b} a été temporairement bloqué. Vous devez vérifier votre identité sous "
            f"{rng.choice(DELAIS)}, faute de quoi votre compte sera définitivement "
            f"suspendu. Confirmez votre mot de passe ici : {url}")

def _p_banque(b, url, name, rng):
    return (f"{b} : nouvelle connexion non autorisée",
            f"Cher utilisateur, une nouvelle connexion à votre compte a été détectée "
            f"depuis {rng.choice(VILLES)}. Si ce n'était pas vous, votre compte est en "
            f"danger. Sécurisez-le immédiatement : {url}. Sans action sous "
            f"{rng.choice(DELAIS)}, votre compte sera fermé.")

def _p_colis(b, url, name, rng):
    return (f"{b} : votre colis est en attente",
            f"Cher client, votre colis n'a pas pu être livré car des frais de douane de "
            f"{_montant(rng)} restent impayés. Veuillez confirmer vos informations et "
            f"payer maintenant pour libérer votre colis, sinon il sera retourné : {url}")

def _p_impots(b, url, name, rng):
    return ("Remboursement d'impôts en attente",
            f"Cher contribuable, vous êtes éligible à un remboursement de {_montant(rng)}. "
            f"Pour le recevoir, vous devez confirmer vos coordonnées bancaires sous "
            f"{rng.choice(DELAIS)} : {url}. Passé ce délai, votre demande sera annulée.")

def _p_motdepasse(b, url, name, rng):
    return (f"Votre mot de passe {b} expire {rng.choice(DELAIS)}",
            f"Cher utilisateur, pour des raisons de sécurité votre mot de passe va "
            f"expirer. Pour conserver l'accès à votre compte, vous devez vous connecter "
            f"et mettre à jour vos identifiants immédiatement : {url}")

PHISH_BUILDERS = [_p_compte, _p_banque, _p_colis, _p_impots, _p_motdepasse]


# ----------------------------------------------------------------- helpers ----
def _generate(builders, brand_pool, label, n, seed, *, nolink=None, phishing=False):
    rng = random.Random(seed)
    seen, rows = set(), []
    attempts = 0
    while len(rows) < n and attempts < n * 40:
        attempts += 1
        build = rng.choice(builders)
        name = rng.choice(PRENOMS)
        if phishing:
            brand = rng.choice(CIBLES)
            subject, body = build(brand, _url_mal(brand, rng), name, rng)
        else:
            brand, dom = rng.choice(brand_pool)
            subject, body = build(brand, dom, name, rng)
        text = clean_text(f"{subject} {body}")
        if text in seen:
            continue
        seen.add(text)
        rows.append(text)
    return pd.DataFrame({"text": rows, "label": label})


def generate_french_legit_df(n: int = 3000, seed: int = 11) -> pd.DataFrame:
    """~n distinct legitimate French emails (label = 0)."""
    return _generate(LEGIT_BUILDERS, MARQUES, 0, n, seed)


def generate_french_phishing_df(n: int = 2000, seed: int = 13) -> pd.DataFrame:
    """~n distinct French phishing emails with malicious URLs (label = 1)."""
    return _generate(PHISH_BUILDERS, None, 1, n, seed, phishing=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aperçu de l'augmentation française")
    ap.add_argument("--preview", type=int, default=6)
    args = ap.parse_args()
    legit = generate_french_legit_df()
    phish = generate_french_phishing_df()
    print(f"{len(legit)} légitimes FR + {len(phish)} phishing FR générés.\n")
    print("== LÉGITIMES ==")
    for t in legit["text"].head(args.preview):
        print("-", t[:180])
    print("\n== PHISHING ==")
    for t in phish["text"].head(args.preview):
        print("-", t[:180])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
