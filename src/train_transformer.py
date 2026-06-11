"""Step 4 — Advanced (optional) model: DistilBERT fine-tuning.

This is heavier than the baseline and benefits a lot from a GPU. If you have
no GPU or a slow machine, use --small-sample to train on a small subset just to
see the pipeline work end-to-end.

    python src/train_transformer.py # full dataset (slow on CPU)
    python src/train_transformer.py --small-sample # ~2000 emails, quick demo

Output:
    models/transformer/ saved model + tokenizer
    reports/transformer_metrics.json

The baseline (train_baseline.py) remains the reference model used by predict.py
and the Streamlit app. This script is optional.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "processed" / "emails.csv"
OUT_DIR = ROOT / "models" / "transformer"
METRICS_PATH = ROOT / "reports" / "transformer_metrics.json"

MODEL_NAME = "distilbert-base-uncased"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT (optionnel)")
    p.add_argument("--small-sample", action="store_true",
                   help="N'utiliser qu'un petit échantillon (démo rapide).")
    p.add_argument("--sample-size", type=int, default=2000,
                   help="Taille de l'échantillon en mode --small-sample.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=256,
                   help="Longueur max de tokens (128-160 = bien plus rapide sur CPU).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not CSV.exists():
        print(f"! Fichier introuvable: {CSV}")
        print(" Lancez d'abord: python src/download_dataset.py")
        return 1

    # Heavy imports are local so the rest of the project works without torch.
    try:
        import torch
        from datasets import Dataset
        from sklearn.metrics import (accuracy_score, f1_score,
                                     precision_score, recall_score)
        from sklearn.model_selection import train_test_split
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer, Trainer, TrainingArguments)
    except ImportError as exc:
        print(f"! Dépendances manquantes: {exc}")
        print(" Installez: pip install torch transformers datasets evaluate")
        return 1

    df = pd.read_csv(CSV)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip().str.len() > 0]
    df["label"] = df["label"].astype(int)

    if args.small_sample:
        per_class = min(args.sample_size, len(df)) // 2
        parts = [g.sample(min(len(g), per_class), random_state=RANDOM_STATE)
                 for _, g in df.groupby("label")]
        df = (pd.concat(parts)
                .sample(frac=1, random_state=RANDOM_STATE)
                .reset_index(drop=True))
        print(f"Mode --small-sample: {len(df)} emails utilisés.")

    has_gpu = torch.cuda.is_available()
    print(f"GPU disponible: {has_gpu}")
    if not has_gpu and not args.small_sample:
        print(" ! Pas de GPU détecté: l'entraînement complet sera TRÈS lent.")
        print(" Conseil: relancez avec --small-sample.")

    train_df, eval_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE,
    )
    print(f"Train: {len(train_df)} | Eval: {len(eval_df)} (stratifié)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    eval_ds = Dataset.from_pandas(eval_df[["text", "label"]], preserve_index=False)
    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        id2label={0: "legitimate", 1: "phishing"},
        label2id={"legitimate": 0, "phishing": 1},
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision_phishing": precision_score(labels, preds, pos_label=1,
                                                  zero_division=0),
            "recall_phishing": recall_score(labels, preds, pos_label=1,
                                            zero_division=0),
            "f1_phishing": f1_score(labels, preds, pos_label=1, zero_division=0),
        }

    training_args = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        # Prioritize phishing recall when comparing across epochs.
        metric_for_best_model="recall_phishing",
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    # transformers >=4.46 renamed `tokenizer` -> `processing_class`.
    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)

    print("Entraînement DistilBERT...")
    trainer.train()
    metrics = trainer.evaluate()

    print("=" * 60)
    print(" RÉSULTATS TRANSFORMER (classe positive = phishing)")
    print("=" * 60)
    for k in ("eval_accuracy", "eval_precision_phishing",
              "eval_recall_phishing", "eval_f1_phishing"):
        if k in metrics:
            print(f" {k:.<30} {metrics[k]:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"[OK] Modèle sauvegardé: {OUT_DIR}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: round(float(v), 4) for k, v in metrics.items()
             if isinstance(v, (int, float))}
    clean["model"] = "distilbert-base-uncased"
    clean["small_sample"] = bool(args.small_sample)
    METRICS_PATH.write_text(json.dumps(clean, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"[OK] Métriques sauvegardées: {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
