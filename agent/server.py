import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from urllib.parse import parse_qs

from .config import VERIFY_TOKEN
from .translate import preprocess_user_text, translate_from_en
from .intent import classify_intent, IntentLabel
from .catalog import CatalogIndex, SAMPLE_PRODUCTS, Product
from .policies import answer_policy_question
from .fb import send_text, send_carousel


app = FastAPI(title="Messenger AI Agent")
catalog = CatalogIndex(SAMPLE_PRODUCTS)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/webhook")
def verify_webhook(mode: Optional[str] = None, hub_mode: Optional[str] = None,
                   hub_challenge: Optional[str] = None, hub_verify_token: Optional[str] = None,
                   verify_token: Optional[str] = None):
    # Meta uses hub.mode, hub.verify_token, hub.challenge
    # Some proxies may forward as different param names; check multiple keys.
    mode = mode or hub_mode or ""
    token = hub_verify_token or verify_token or ""
    challenge = hub_challenge or ""
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(challenge)
    return PlainTextResponse("Forbidden", status_code=403)


def _handle_text_message(sender_id: str, text: str):
    # Detect and translate to English for processing
    english_text, user_lang = preprocess_user_text(text)

    # Intent classification
    intent = classify_intent(english_text, language=user_lang)

    if intent.intent == IntentLabel.product_search:
        # Exact first
        exact = catalog.find_exact(name=intent.product_name, color=intent.color, category=intent.category)
        products: List[Product] = []
        if exact:
            products = exact[:10]
        else:
            # Similar search using features and query text
            query = english_text
            sims = catalog.find_similar(query, top_k=6)
            products = [p for p, _ in sims]

        if products:
            send_carousel(sender_id, products)
            return
        else:
            reply = "Sorry, I couldn't find matching products."
            send_text(sender_id, translate_from_en(reply, user_lang))
            return

    if intent.intent in {IntentLabel.policy, IntentLabel.shipping, IntentLabel.returns}:
        answer = answer_policy_question(english_text)
        send_text(sender_id, translate_from_en(answer, user_lang))
        return

    if intent.intent == IntentLabel.general_query:
        # Lightweight general response via DeepSeek
        from .hf_client import make_deepseek_model
        model = make_deepseek_model()
        result = model.invoke(english_text)
        send_text(sender_id, translate_from_en(result.content, user_lang))
        return

    # Fallback unknown
    send_text(sender_id, translate_from_en("I didn't catch that. How can I help?", user_lang))


@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()
    logging.info("Incoming webhook: %s", body)

    # Standard Messenger structure: entry -> messaging[]
    for entry in body.get("entry", []):
        for messaging_event in entry.get("messaging", []):
            sender = messaging_event.get("sender", {}).get("id")
            if not sender:
                continue
            message = messaging_event.get("message")
            if message and "text" in message:
                _handle_text_message(sender, message["text"]) 
            # Optionally handle postbacks, quick replies, etc.
    return JSONResponse({"status": "ok"})


# Local dev runner: uvicorn agent.server:app --reload --port 8000
