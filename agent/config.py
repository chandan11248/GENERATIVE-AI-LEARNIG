import os
from dotenv import load_dotenv

load_dotenv()

PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN", "")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")

# Hugging Face DeepSeek model (override with env var if needed)
HF_DEEPSEEK_REPO_ID = os.getenv("HF_DEEPSEEK_REPO_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

# Placeholder image for products lacking images
PLACEHOLDER_IMAGE = os.getenv(
    "PLACEHOLDER_IMAGE",
    "https://via.placeholder.com/600x400?text=No+Image",
)

# Buy URL base for CTA buttons
BUY_BASE_URL = os.getenv("BUY_BASE_URL", "https://example.com/buy")
