# PhishMail Detector

Détecteur de phishing par apprentissage automatique. Le projet va d'un modèle
simple (TF-IDF + régression logistique) à un modèle plus avancé et optionnel
(DistilBERT). Il donne un verdict, un score de risque et des explications, et il
y a une petite interface web faite avec Streamlit.

Pour le détail de la démarche, des problèmes rencontrés et des méthodes, voir le
fichier [`RAPPORT.md`](./RAPPORT.md).

Attention : c'est un outil éducatif. Il peut se tromper et il ne remplace pas une
vraie passerelle de sécurité email. Ne vous fiez jamais uniquement à ce score
pour une décision réelle.

## Objectif

Classer un email comme phishing (1) ou légitime (0), afficher un score de risque
(0-100 %) et des explications lisibles. La priorité choisie est le rappel sur le
phishing : mieux vaut une fausse alerte de trop que de laisser passer un email
dangereux.

## Structure du projet

```
PhishingDetector/
├── app.py                       # Interface Streamlit
├── requirements.txt
├── RAPPORT.md                   # Rapport détaillé du projet
├── data/
│   ├── raw/                     # Datasets bruts (non versionnés)
│   └── processed/emails.csv     # Données normalisées : text,label
├── models/
│   ├── baseline_tfidf_logreg.joblib
│   └── transformer/             # DistilBERT (optionnel)
├── reports/                     # Métriques et logs d'expériences
└── src/
    ├── download_dataset.py      # 1. récupération + nettoyage
    ├── analyze_dataset.py       # 2. qualité du dataset
    ├── train_baseline.py        # 3. TF-IDF + LogReg (modèle de base)
    ├── train_url_classifier.py  # 4. sous-modèle d'URL
    ├── train_meta.py            # 5. méta-classifieur
    ├── train_transformer.py     # 6. DistilBERT (optionnel)
    └── predict.py               # prédiction + explications
```

Le format des données est `data/processed/emails.csv` avec deux colonnes `text`
et `label` (0 = légitime, 1 = phishing).

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Pour la partie de base et l'interface, `torch`, `transformers` et `evaluate` ne
sont pas indispensables ; ils ne servent que pour l'étape DistilBERT.

## Utilisation

Télécharger le dataset :

```bash
python src/download_dataset.py
```

Analyser la qualité du dataset :

```bash
python src/analyze_dataset.py
```

Entraîner le modèle de base :

```bash
python src/train_baseline.py
```

Faire une prédiction en ligne de commande :

```bash
python src/predict.py --text "URGENT: verify your bank password now http://x.tk"
python src/predict.py --file mon_email.txt
```

La sortie est un JSON avec le verdict, le score de risque, le modèle utilisé et
les explications.

Lancer l'interface web :

```bash
streamlit run app.py
```

L'interface permet de coller un email, d'avoir un verdict, un score en
pourcentage et des explications. Toute l'analyse se fait en local : aucun email
n'est envoyé vers une API externe et rien n'est stocké.

## Résultats

Quelques chiffres obtenus sur la partie test (voir `RAPPORT.md` et `reports/`
pour le détail) :

| Métrique | Modèle de base | Méta-classifieur |
|----------|----------------|------------------|
| Accuracy / AUC | 0.99 | 0.999 |
| Précision (phishing) | 0.99 | 0.99 |
| Rappel (phishing) | 0.99 | 0.99 |

## Limites

- Les données sont surtout en anglais, donc le modèle est moins bon dans les
  autres langues.
- Le modèle raisonne sur les mots, il peut être trompé par des attaques
  nouvelles.
- Il reste des faux positifs et des faux négatifs.

## Confidentialité

- Analyse 100 % locale, aucun envoi d'email vers une API externe.
- Pas de clés API dans le dépôt : utiliser des variables d'environnement
  (`HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`).
- `data/raw/`, `data/processed/` et les modèles ne sont pas versionnés (voir
  `.gitignore`).
