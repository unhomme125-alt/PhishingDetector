"""Shared text helpers.

IMPORTANT: we deliberately KEEP urls in the text. They are one of the strongest
phishing signals, so removing them would throw away information.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def strip_html(text: str) -> str:
    """Remove HTML tags but keep the visible text (and any urls inside it)."""
    if "<" not in text or ">" not in text:
        return text
    try:
        return BeautifulSoup(text, "html.parser").get_text(separator=" ")
    except Exception:
        return text


def clean_text(text: str) -> str:
    """Light, non-destructive cleaning.

    - decode HTML to plain text
    - collapse whitespace
    We do NOT strip urls, digits or punctuation: they carry signal.
    """
    if text is None:
        return ""
    text = str(text)
    text = strip_html(text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_urls(text: str) -> list[str]:
    """Return all urls found in the text."""
    if not text:
        return []
    return URL_RE.findall(str(text))
