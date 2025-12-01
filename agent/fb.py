import logging
import requests
from typing import List
from .config import PAGE_ACCESS_TOKEN, BUY_BASE_URL
from .catalog import Product, product_to_messenger_element


FB_SEND_URL = "https://graph.facebook.com/v17.0/me/messages"


def _send(payload: dict):
    if not PAGE_ACCESS_TOKEN:
        logging.info("[FB SEND MOCK] %s", payload)
        return {"mock": True}
    params = {"access_token": PAGE_ACCESS_TOKEN}
    r = requests.post(FB_SEND_URL, params=params, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


def send_text(recipient_id: str, text: str):
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": text[:2000]},  # FB limit safeguard
    }
    return _send(payload)


def format_carousel(products: List[Product]) -> dict:
    elements = []
    for p in products[:10]:  # FB generic template limit
        el = product_to_messenger_element(p)
        el["buttons"] = [
            {
                "type": "web_url",
                "url": f"{BUY_BASE_URL}?sku={p.sku}",
                "title": "Buy",
            }
        ]
        elements.append(el)
    return {
        "attachment": {
            "type": "template",
            "payload": {
                "template_type": "generic",
                "elements": elements,
            },
        }
    }


def send_carousel(recipient_id: str, products: List[Product]):
    payload = {
        "recipient": {"id": recipient_id},
        "message": format_carousel(products),
    }
    return _send(payload)
