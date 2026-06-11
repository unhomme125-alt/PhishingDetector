"""Step 2 — Dataset quality evaluation.

Reads data/processed/emails.csv and prints quality stats plus a "solidity"
score out of 100, broken down into weighted criteria:

    taille suffisante .......... /30
    équilibre des classes ...... /25
    peu de doublons ............ /20
    peu de valeurs vides ....... /15
    longueur texte exploitable . /10
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "processed" / "emails.csv"


def score_size(n: int) -> float:
    """/30 — full marks at >= 10k emails, partial below."""
    if n >= 10_000:
        return 30.0
    if n >= 5_000:
        return 25.0
    if n >= 2_000:
        return 18.0
    if n >= 500:
        return 10.0
    if n >= 100:
        return 4.0
    return 0.0


def score_balance(pct_phish: float) -> float:
    """/25 — best near 50/50, scaled by distance from balance."""
    # distance from 50% (0 when perfectly balanced, 0.5 when all one class)
    distance = abs(pct_phish - 50.0) / 50.0
    return round(max(0.0, 1.0 - distance) * 25.0, 1)


def score_duplicates(dup_ratio: float) -> float:
    """/20 — fewer duplicates is better."""
    return round(max(0.0, 1.0 - dup_ratio) * 20.0, 1)


def score_empty(empty_ratio: float) -> float:
    """/15 — fewer empty texts is better."""
    return round(max(0.0, 1.0 - empty_ratio) * 15.0, 1)


def score_length(avg_len: float) -> float:
    """/10 — usable average length (~chars). Full marks around 200+ chars."""
    if avg_len >= 200:
        return 10.0
    if avg_len >= 100:
        return 7.0
    if avg_len >= 50:
        return 4.0
    if avg_len >= 20:
        return 2.0
    return 0.0


def verdict(score: float) -> str:
    if score >= 80:
        return "solide pour un premier modèle"
    if score >= 60:
        return "correct, utilisable avec prudence"
    if score >= 40:
        return "fragile, à améliorer avant de conclure"
    return "insuffisant, ne pas se fier aux résultats"


def main() -> int:
    if not CSV.exists():
        print(f"! Fichier introuvable: {CSV}")
        print(" Lancez d'abord: python src/download_dataset.py")
        return 1

    df = pd.read_csv(CSV)
    df["text"] = df["text"].fillna("").astype(str)

    total = len(df)
    n_phish = int((df["label"] == 1).sum())
    n_legit = int((df["label"] == 0).sum())
    pct_phish = (n_phish / total * 100) if total else 0.0
    pct_legit = (n_legit / total * 100) if total else 0.0

    lengths = df["text"].str.len()
    avg_len = float(lengths.mean()) if total else 0.0

    n_dup = int(df.duplicated(subset=["text"]).sum())
    dup_ratio = (n_dup / total) if total else 0.0

    n_empty = int((df["text"].str.strip().str.len() == 0).sum())
    empty_ratio = (n_empty / total) if total else 0.0

    # --- weighted solidity score ---
    parts = {
        "taille suffisante (/30)": score_size(total),
        "équilibre des classes (/25)": score_balance(pct_phish),
        "peu de doublons (/20)": score_duplicates(dup_ratio),
        "peu de valeurs vides (/15)": score_empty(empty_ratio),
        "longueur texte exploitable (/10)": score_length(avg_len),
    }
    solidity = round(sum(parts.values()))

    print("=" * 60)
    print(" ANALYSE DU DATASET")
    print("=" * 60)
    print(f" Nombre total d'emails .......... {total}")
    print(f" Phishing ....................... {n_phish} ({pct_phish:.1f}%)")
    print(f" Légitimes ...................... {n_legit} ({pct_legit:.1f}%)")
    print(f" Longueur moyenne (caractères) .. {avg_len:.1f}")
    print(f" Doublons (texte) ............... {n_dup} ({dup_ratio*100:.1f}%)")
    print(f" Emails vides ................... {n_empty} ({empty_ratio*100:.1f}%)")
    print("-" * 60)
    print(" DÉTAIL DU SCORE")
    for name, value in parts.items():
        print(f" {name:.<40} {value}")
    print("-" * 60)
    print(f"Dataset solidity score: {solidity}% - {verdict(solidity)}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
