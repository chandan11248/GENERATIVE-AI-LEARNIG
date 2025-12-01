from typing import Tuple
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

# Make langdetect deterministic
DetectorFactory.seed = 0


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"


def translate_to_en(text: str, src_lang: str) -> str:
    if src_lang.startswith("en"):
        return text
    try:
        return GoogleTranslator(source=src_lang, target="en").translate(text)
    except Exception:
        return text


def translate_from_en(text: str, target_lang: str) -> str:
    if target_lang.startswith("en"):
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception:
        return text


def preprocess_user_text(text: str) -> Tuple[str, str]:
    lang = detect_language(text)
    english = translate_to_en(text, lang)
    return english, lang
