"""Step 1 — Dataset retrieval and normalization.

Priority order:
  1. Realistic email corpus (GitHub mirror of naserabdullahalam's
     phishing-email-dataset: CEAS_08, Nazario, Nigerian_Fraud, SpamAssassin,
     Enron, Ling). This is the RECOMMENDED source: it mixes real modern
     phishing (account/credential/banking lures) with diverse legitimate
     business email, so the model learns phishing signals instead of a
     spam-vs-academic vocabulary shortcut.
  2. Hugging Face : ealvaradob/phishing-dataset (texts.json) — fallback.
     NOTE: this split is essentially old spam-vs-academic email; it scores
     well on its own test set but generalizes poorly to real inboxes.
  3. Kaggle : naserabdullahalam/phishing-email-dataset (needs creds).
  4. Manual download instructions (printed if all fail).

Output:
  data/raw/ raw downloaded files (kept as-is, not versioned)
  data/processed/emails.csv normalized: columns [text, label]
                              label: 0 = legitimate, 1 = phishing

We KEEP urls in the text on purpose (strong phishing signal).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running as `python src/download_dataset.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import clean_text # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_CSV = PROCESSED_DIR / "emails.csv"

# Realistic email corpus — public GitHub mirror of the Kaggle
# naserabdullahalam/phishing-email-dataset (no credentials required).
# Each file has [subject, body, label] (some also [sender, receiver, urls]).
GITHUB_BASE = (
    "https://raw.githubusercontent.com/"
    "rokibulroni/Phishing-Email-Dataset/main/"
)
GITHUB_FILES = [
    "CEAS_08.csv", # large mixed phishing/spam + mailing-list ham
    "Nazario.csv", # real phishing inbox (account/credential lures)
    "Nigerian_Fraud.csv", # advance-fee fraud
    "SpamAssasin.csv", # classic spam + ham
    "Enron.csv", # real business email (diverse legitimate)
    "Ling.csv", # ling-spam corpus
]
# Cap absurdly long emails (a few rows reach millions of chars) so TF-IDF and
# the tokenizer stay fast; the head of an email holds the signal anyway.
MAX_TEXT_CHARS = 20_000

# Optional, transparent augmentation: legitimate transactional / service
# notification emails (see src/generate_legit_transactional.py). The real
# corpus lacks these, so legit "your account / sign in / statement" wording is
# wrongly learned as phishing. Set to 0 to disable the augmentation entirely.
SYNTHETIC_LEGIT_N = 2800

# Symmetric augmentation: modern phishing emails with realistic malicious URLs
# (lookalike domains, suspicious TLDs, IP hosts — see
# src/generate_phishing_url.py). The real corpus's phishing URLs look ordinary,
# so the URL submodel can't learn modern URL tricks. Set to 0 to disable.
SYNTHETIC_PHISH_N = 2500

# Language augmentation: FRENCH emails in BOTH classes (see src/generate_french.py).
# The corpus + other augmentations are ~100% English, so French legitimate email
# is flagged as phishing. Both classes are added so French is learned
# discriminatively, not as "French = legitimate". Set to 0 to disable.
FRENCH_LEGIT_N = 3000
FRENCH_PHISH_N = 2000

# Common column names seen across phishing datasets.
TEXT_CANDIDATES = ["text", "body", "email", "content", "message", "Email Text", "email_text"]
LABEL_CANDIDATES = ["label", "labels", "class", "type", "Email Type", "target", "is_phishing"]

# Maps various textual labels to our convention (0 legit / 1 phishing).
LABEL_MAP = {
    "phishing": 1, "phish": 1, "spam": 1, "fraud": 1, "malicious": 1,
    "phishing email": 1, "1": 1, "true": 1, "yes": 1,
    "legitimate": 0, "legit": 0, "ham": 0, "safe": 0, "normal": 0, "benign": 0,
    "safe email": 0, "0": 0, "false": 0, "no": 0,
}

MANUAL_INSTRUCTIONS = """
============================================================
 Téléchargement automatique impossible.
 Téléchargez le dataset manuellement, puis relancez ce script.
------------------------------------------------------------
 OPTION A — Hugging Face (recommandé)
   1. pip install datasets
   2. (si besoin) export HF_TOKEN=... / set HF_TOKEN=...
   3. relancez : python src/download_dataset.py

 OPTION B — Kaggle
   1. pip install kaggle
   2. Créez un token Kaggle : https://www.kaggle.com/settings (Create New Token)
   3. Placez kaggle.json dans ~/.kaggle/ (ou %USERPROFILE%\\.kaggle\\)
   4. kaggle datasets download -d naserabdullahalam/phishing-email-dataset \\
        -p data/raw --unzip
   5. relancez : python src/download_dataset.py

 OPTION C — Dépôt manuel
   Déposez n'importe quel CSV avec une colonne texte et une colonne label
   dans data/raw/ puis relancez ce script. Colonnes texte reconnues :
     {text}
   Colonnes label reconnues :
     {label}
============================================================
""".format(text=", ".join(TEXT_CANDIDATES), label=", ".join(LABEL_CANDIDATES))


def _pick_column(columns, candidates):
    """Case-insensitive match of a dataframe column against candidates."""
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _map_label(value) -> int | None:
    """Convert a raw label into 0/1, or None if unmappable."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return int(value)
        # some datasets use -1/1 or 1/2 ; treat the larger value as phishing
        return 1 if value > 0 else 0
    key = str(value).strip().lower()
    return LABEL_MAP.get(key)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    """Reduce an arbitrary dataframe to columns [text, label]."""
    text_col = _pick_column(df.columns, TEXT_CANDIDATES)
    label_col = _pick_column(df.columns, LABEL_CANDIDATES)
    if text_col is None or label_col is None:
        print(f" ! Colonnes non reconnues: {list(df.columns)}")
        return None

    out = pd.DataFrame({
        "text": df[text_col].map(clean_text),
        "label": df[label_col].map(_map_label),
    })
    out = out.dropna(subset=["label"])
    out = out[out["text"].str.len() > 0]
    out["label"] = out["label"].astype(int)
    return out


def _build_text(df: pd.DataFrame) -> pd.Series:
    """Combine subject + body into a single cleaned text column.

    URLs are kept (clean_text is non-destructive). Very long emails are
    truncated to MAX_TEXT_CHARS.
    """
    subject = df["subject"].fillna("") if "subject" in df.columns else ""
    body = df["body"].fillna("") if "body" in df.columns else df.get("text", "")
    combined = (subject.astype(str) + " " + body.astype(str)).str.slice(0, MAX_TEXT_CHARS)
    return combined.map(clean_text)


def try_github_realistic() -> pd.DataFrame | None:
    """Download and combine the realistic email corpus (priority source).

    Returns a normalized [text, label] frame, or None if nothing could be
    fetched (e.g. no internet).
    """
    import io
    import urllib.request

    print("-> Source réaliste (miroir GitHub: CEAS_08, Nazario, Nigerian_Fraud, "
          "SpamAssassin, Enron, Ling)")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for fn in GITHUB_FILES:
        try:
            req = urllib.request.Request(
                GITHUB_BASE + fn, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=120).read()
        except Exception as exc: # noqa: BLE001
            print(f" ! {fn}: téléchargement impossible ({exc})")
            continue
        (RAW_DIR / fn).write_bytes(data) # keep raw copy (not versioned)
        try:
            df = pd.read_csv(io.BytesIO(data))
        except Exception as exc: # noqa: BLE001
            print(f" ! {fn}: lecture impossible ({exc})")
            continue
        if "label" not in df.columns:
            print(f" ! {fn}: pas de colonne 'label' — ignoré")
            continue
        part = pd.DataFrame({
            "text": _build_text(df),
            "label": df["label"].map(_map_label),
        })
        part = part.dropna(subset=["label"])
        part = part[part["text"].str.len() > 0]
        part["label"] = part["label"].astype(int)
        n1 = int((part["label"] == 1).sum())
        print(f" [OK] {fn}: {len(part)} emails (phishing={n1}, légitimes={len(part)-n1})")
        frames.append(part)

    if not frames:
        print(" ! Aucune source réaliste récupérée.")
        return None
    return pd.concat(frames, ignore_index=True)


def try_huggingface() -> pd.DataFrame | None:
    """Download the 'texts' split of ealvaradob/phishing-dataset.

    The repo ships a (now-unsupported) loading script, so instead of
    `load_dataset(...)` we pull the raw `texts.json` file directly from the
    hub. That file already has columns [text, label] with label in {0, 1}.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(" ! `huggingface_hub` non installé — on saute Hugging Face.")
        return None

    print("-> Tentative Hugging Face: ealvaradob/phishing-dataset (texts.json)")
    try:
        path = hf_hub_download(
            repo_id="ealvaradob/phishing-dataset",
            filename="texts.json",
            repo_type="dataset",
        )
        df = pd.read_json(path)
    except Exception as exc: # noqa: BLE001
        print(f" ! Échec Hugging Face: {exc}")
        return None

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_DIR / "huggingface_texts.parquet", index=False)
    print(f" [OK] {len(df)} lignes récupérées depuis Hugging Face.")
    return normalize_frame(df)


def try_kaggle() -> pd.DataFrame | None:
    """Attempt to download the Kaggle phishing-email-dataset."""
    try:
        import kaggle # noqa: F401
    except (ImportError, OSError) as exc:
        print(f" ! Kaggle indisponible ({exc}) — on saute Kaggle.")
        return None

    print("-> Tentative Kaggle: naserabdullahalam/phishing-email-dataset")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(
            "naserabdullahalam/phishing-email-dataset",
            path=str(RAW_DIR), unzip=True,
        )
    except Exception as exc: # noqa: BLE001
        print(f" ! Échec Kaggle: {exc}")
        return None

    return load_from_raw()


def load_from_raw() -> pd.DataFrame | None:
    """Read any CSV/parquet already present in data/raw/ and normalize it."""
    if not RAW_DIR.exists():
        return None
    files = list(RAW_DIR.glob("*.csv")) + list(RAW_DIR.glob("*.parquet"))
    if not files:
        return None

    frames = []
    for f in files:
        print(f"-> Lecture locale: {f.name}")
        try:
            df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        except Exception as exc: # noqa: BLE001
            print(f" ! Lecture impossible ({exc})")
            continue
        norm = normalize_frame(df)
        if norm is not None and len(norm) > 0:
            frames.append(norm)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Preferred: the realistic combined email corpus (fresh download).
    df = try_github_realistic()

    # 2. Offline fallback: reuse whatever is already in data/raw/.
    if df is None or len(df) == 0:
        df = load_from_raw()

    # 3. Last resorts (weaker / need credentials).
    if df is None or len(df) == 0:
        df = try_huggingface()
    if df is None or len(df) == 0:
        df = try_kaggle()

    if df is None or len(df) == 0:
        print(MANUAL_INSTRUCTIONS)
        return 1

    # 4. Optional augmentation: legitimate transactional / notification emails
    # to cover a known blind spot in the real corpus (reduces false
    # positives on legit "your account / sign in / statement" wording).
    if SYNTHETIC_LEGIT_N > 0:
        try:
            from generate_legit_transactional import generate_legit_df
            synth = generate_legit_df(n=SYNTHETIC_LEGIT_N)
            print(f" + augmentation légitime transactionnelle (synthétique): "
                  f"{len(synth)} emails")
            df = pd.concat([df, synth], ignore_index=True)
        except Exception as exc: # noqa: BLE001
            print(f" ! Augmentation légitime ignorée ({exc})")

    # 5. Symmetric augmentation: modern phishing emails with malicious URLs, to
    # cover the corpus blind spot where phishing URLs look ordinary (lets the
    # URL submodel learn suspicious TLDs / lookalike domains / IP hosts).
    if SYNTHETIC_PHISH_N > 0:
        try:
            from generate_phishing_url import generate_phishing_df
            synth_p = generate_phishing_df(n=SYNTHETIC_PHISH_N)
            print(f" + augmentation phishing (URLs malveillantes, synthétique): "
                  f"{len(synth_p)} emails")
            df = pd.concat([df, synth_p], ignore_index=True)
        except Exception as exc: # noqa: BLE001
            print(f" ! Augmentation phishing ignorée ({exc})")

    # 6. Language augmentation: French emails (both classes) so French legitimate
    # email stops being flagged as phishing (out-of-distribution language).
    if FRENCH_LEGIT_N > 0 or FRENCH_PHISH_N > 0:
        try:
            from generate_french import (generate_french_legit_df,
                                          generate_french_phishing_df)
            fr_legit = generate_french_legit_df(n=FRENCH_LEGIT_N)
            fr_phish = generate_french_phishing_df(n=FRENCH_PHISH_N)
            print(f" + augmentation française (synthétique): "
                  f"{len(fr_legit)} légitimes + {len(fr_phish)} phishing")
            df = pd.concat([df, fr_legit, fr_phish], ignore_index=True)
        except Exception as exc: # noqa: BLE001
            print(f" ! Augmentation française ignorée ({exc})")

    # Final cleanup: drop empties and exact duplicates.
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip().str.len() > 0]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)
    n_phish = int((df["label"] == 1).sum())
    n_legit = int((df["label"] == 0).sum())
    print("------------------------------------------------------------")
    print(f"[OK] Dataset normalisé écrit dans: {OUTPUT_CSV}")
    print(f" Total: {len(df)} | phishing: {n_phish} | légitimes: {n_legit}")
    print(" Colonnes: text, label (0=legitimate, 1=phishing)")
    print("------------------------------------------------------------")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
