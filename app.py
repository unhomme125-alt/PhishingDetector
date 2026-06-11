"""Step 6 — Streamlit test interface for PhishMail Detector.

Run with: streamlit run app.py

Privacy: everything runs locally. No email is sent to any external API, and
nothing is stored unless you explicitly tick the consent box.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predict import load_model, predict_email # noqa: E402

# --- prefilled examples -----------------------------------------------------
EXAMPLES = {
    "— Choisir un exemple —": "",
    "Phishing bancaire": (
        "Dear customer,\n\n"
        "We detected unusual activity on your bank account and it has been "
        "temporarily BLOCKED. To restore access you must verify your identity "
        "and password within 24 hours, otherwise your account will be "
        "permanently suspended.\n\n"
        "Verify now: http://secure-bank-verify.tk/login\n\n"
        "Bank Security Team"
    ),
    "Faux colis": (
        "Your parcel could not be delivered.\n\n"
        "A customs fee of 1.99 EUR is required to release your package. "
        "Please confirm your delivery details and pay now to avoid return:\n"
        "https://bit.ly/track-parcel-payment\n\n"
        "Delivery Service"
    ),
    "Email légitime étudiant": (
        # NB: les datasets publics sont surtout anglophones, le modèle classe
        # donc plus fiablement l'anglais (voir 'Limites' du README).
        "Hi everyone,\n\n"
        "Just a reminder that our study group for the chemistry midterm meets "
        "on Thursday at 6pm in the library, room 204. Bring your lecture notes "
        "from weeks 3 to 5 and we'll go through the practice problems together.\n\n"
        "See you there!\nTom"
    ),
    "Email légitime professionnel": (
        "Hi team,\n\n"
        "Please find attached the agenda for our quarterly review on Friday. "
        "Let me know if you'd like to add any topics before Thursday.\n\n"
        "Best regards,\nSarah"
    ),
}


def badge(verdict: str, risk: float) -> None:
    """Color badge driven by the cascade verdict (phishing/suspect/legitimate)."""
    if verdict == "phishing":
        color, label = "#c0392b", "PHISHING" # red
    elif verdict == "suspect":
        color, label = "#e67e22", "SUSPECT" # orange
    elif verdict == "indéterminé":
        color, label = "#7f8c8d", "INDÉTERMINÉ" # grey
    else:
        color, label = "#27ae60", "LÉGITIME" # green

    st.markdown(
        f"""
        <div style="background:{color};color:white;padding:16px;border-radius:10px;
                    text-align:center;font-size:24px;font-weight:bold;">
            {label} — {risk:.1f}%
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="PhishMail Detector", page_icon="")
    st.title(" PhishMail Detector")
    st.caption(
        "Outil **éducatif** de détection de phishing. Analyse 100% locale — "
        "aucun email n'est envoyé vers une API externe."
    )

    # Load model once (cached by predict.load_model).
    try:
        load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    choice = st.selectbox("Exemples préremplis", list(EXAMPLES.keys()))
    default_text = EXAMPLES[choice]

    email_text = st.text_area(
        "Collez le contenu de l'email ici",
        value=default_text,
        height=260,
        placeholder="Sujet + corps de l'email...",
    )

    analyze = st.button(" Analyser", type="primary")

    if analyze:
        if not email_text.strip():
            st.warning("Veuillez coller un email à analyser.")
            return

        result = predict_email(email_text)
        st.subheader("Résultat")
        badge(result["verdict"], result["risk_score"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Verdict", result["verdict"])
        col2.metric("Score de risque", f"{result['risk_score']:.1f}%")
        col3.metric("Pré-filtre (Stage 1)", f"{result.get('stage1_score', 0.0):.1f}%")
        st.progress(min(1.0, result["risk_score"] / 100.0))

        st.subheader("Explications")
        for reason in result["explanations"]:
            st.write(f"- {reason}")

        st.caption(f"Modèle: `{result['model']}`")

    with st.expander(" Avertissement"):
        st.write(
            "Cet outil est un démonstrateur éducatif. Il peut se tromper "
            "(faux positifs et faux négatifs) et **ne remplace pas** une vraie "
            "passerelle de sécurité email. Ne vous fiez jamais uniquement à ce "
            "score pour une décision réelle."
        )


if __name__ == "__main__":
    main()
