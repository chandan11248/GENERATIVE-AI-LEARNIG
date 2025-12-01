from enum import Enum
from typing import List, Optional
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from .hf_client import make_deepseek_model


class IntentLabel(str, Enum):
    product_search = "product_search"
    general_query = "general_query"
    policy = "policy"
    shipping = "shipping"
    returns = "returns"
    unknown = "unknown"


class IntentPayload(BaseModel):
    intent: IntentLabel
    product_name: Optional[str] = None
    color: Optional[str] = None
    category: Optional[str] = None
    features: Optional[List[str]] = None
    language: Optional[str] = None


prompt = ChatPromptTemplate.from_template(
    """
You are an intent classifier for an e-commerce chat. Classify the user's message.
Return a JSON strictly matching the provided schema fields.

Intents:
- product_search: Searching or asking for products.
- general_query: General questions not tied to store policies.
- policy: Questions about store policies.
- shipping: Questions about delivery time, cost, regions.
- returns: Questions about returns/refunds/exchanges.
- unknown: Anything else.

Extract product name, color, category and features if present.
User message: {text}
Language (if known): {language}
"""
)


def classify_intent(text: str, language: Optional[str] = None) -> IntentPayload:
    model = make_deepseek_model()
    schema = IntentPayload
    structured = model.with_structured_output(schema)
    chain = prompt | structured
    return chain.invoke({"text": text, "language": language or "unknown"})
