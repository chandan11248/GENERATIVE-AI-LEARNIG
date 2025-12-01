# Messenger AI Agent (DeepSeek R1 via Hugging Face)

This lightweight agent receives Facebook Messenger messages via webhook, classifies intent with DeepSeek R1, searches a small product catalog (exact/similar), sends product carousels or text answers, and supports multi-language via auto-translate.

## Features
- Receives Messenger messages (GET verify, POST events)
- Intent extraction (product_search, general_query, policy, shipping, returns, unknown)
- Exact and similar product search (embeddings + cosine similarity)
- Generic template carousel with images, price, stock, Buy buttons
- Placeholder images when missing
- Policy Q&A grounded on snippets
- Multilingual: auto-detect + translate to English for AI, translate replies back

## Files
- `agent/server.py`: FastAPI app with `/webhook` and `/health`
- `agent/hf_client.py`: DeepSeek (HF) chat client factory
- `agent/intent.py`: Structured intent classification (Pydantic)
- `agent/catalog.py`: In-memory sample catalog + embedding index
- `agent/translate.py`: Language detection + translation helpers
- `agent/fb.py`: Messenger send helpers (text, carousel)
- `agent/policies.py`: Simple policy QA using DeepSeek
- `agent/config.py`: Env vars and defaults

## Setup
1. Install deps:
```zsh
pip install -r requirement.txt
```
2. Add `.env` keys:
```
VERIFY_TOKEN=your_webhook_verify_token
PAGE_ACCESS_TOKEN=your_page_access_token
HF_TOKEN=your_huggingface_token_optional
HF_DEEPSEEK_REPO_ID=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
BUY_BASE_URL=https://example.com/buy
```

## Run
```zsh
uvicorn agent.server:app --reload --port 8000
```
Configure your FB webhook to `https://<your-host>/webhook` with the same `VERIFY_TOKEN`.

## Notes
- If `PAGE_ACCESS_TOKEN` is unset, send calls are logged (mock mode) for local testing.
- Update `SAMPLE_PRODUCTS` in `agent/catalog.py` to match your inventory source.
- For larger catalogs, persist embeddings or use a vector DB.
- DeepSeek R1 outputs can be verbose; keep prompts concise and use structured outputs.