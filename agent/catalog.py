from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from .config import PLACEHOLDER_IMAGE


@dataclass
class Product:
    sku: str
    title: str
    category: str
    color: Optional[str]
    features: List[str]
    price: float
    image_url: Optional[str]
    stock: int


SAMPLE_PRODUCTS: List[Product] = [
    Product(
        sku="TSHIRT-RED-001",
        title="Classic T-Shirt",
        category="apparel",
        color="red",
        features=["cotton", "unisex", "casual"],
        price=19.99,
        image_url=None,
        stock=42,
    ),
    Product(
        sku="TSHIRT-BLU-001",
        title="Classic T-Shirt",
        category="apparel",
        color="blue",
        features=["cotton", "unisex", "casual"],
        price=19.99,
        image_url="https://picsum.photos/seed/tshirt-blue/600/400",
        stock=5,
    ),
    Product(
        sku="SNEAK-WHT-123",
        title="Lightweight Sneakers",
        category="footwear",
        color="white",
        features=["breathable", "lightweight", "sport"],
        price=59.99,
        image_url="https://picsum.photos/seed/sneaker-white/600/400",
        stock=12,
    ),
    Product(
        sku="SNEAK-BLK-124",
        title="Lightweight Sneakers",
        category="footwear",
        color="black",
        features=["breathable", "lightweight", "sport"],
        price=59.99,
        image_url=None,
        stock=0,
    ),
    Product(
        sku="MUG-CRM-900",
        title="Ceramic Coffee Mug",
        category="home",
        color="cream",
        features=["dishwasher-safe", "12oz", "gift"],
        price=9.99,
        image_url="https://picsum.photos/seed/mug-cream/600/400",
        stock=77,
    ),
]


class CatalogIndex:
    def __init__(self, products: List[Product]):
        self.products = products
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectors = self._build_vectors(products)

    def _product_text(self, p: Product) -> str:
        feats = ", ".join(p.features)
        return f"{p.title} category:{p.category} color:{p.color or ''} features:{feats}"

    def _build_vectors(self, products: List[Product]) -> np.ndarray:
        texts = [self._product_text(p) for p in products]
        vecs = np.array(self.embeddings.embed_documents(texts))
        return vecs

    def find_exact(self, name: Optional[str] = None, color: Optional[str] = None, category: Optional[str] = None) -> List[Product]:
        results = self.products
        if name:
            results = [p for p in results if name.lower() in p.title.lower()]
        if color:
            results = [p for p in results if p.color and color.lower() == p.color.lower()]
        if category:
            results = [p for p in results if p.category.lower() == category.lower()]
        return results

    def find_similar(self, query: str, top_k: int = 4) -> List[Tuple[Product, float]]:
        qv = np.array(self.embeddings.embed_documents([query]))
        sims = cosine_similarity(qv, self.vectors)[0]
        idx_scores = list(enumerate(sims))
        idx_scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in idx_scores[:top_k]:
            results.append((self.products[idx], float(score)))
        return results


def product_to_messenger_element(p: Product) -> dict:
    image = p.image_url or PLACEHOLDER_IMAGE
    subtitle = f"{p.color or 'n/a'} • ${p.price:.2f} • Stock: {p.stock}"
    return {
        "title": p.title,
        "image_url": image,
        "subtitle": subtitle,
        "buttons": [
            # Buttons filled in fb.format_carousel depending on BUY_BASE_URL
        ],
    }
