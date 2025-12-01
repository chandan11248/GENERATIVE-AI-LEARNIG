# Copilot Instructions for GENERATIVE-AI-LEARNIG

These rules help AI coding agents be productive in this repo of LangChain demos. Keep answers concrete, use real file paths, and prefer small, runnable examples.

## Repo Overview
- Purpose: Hands-on scripts showing LangChain patterns across LLM chat, embeddings, prompting, parsing, and structured output.
- Structure highlights:
  - `1.LLM/`: Basic completion via `OpenAI` (instruct).
  - `2.chatModels/`: Chat models via OpenAI, Groq, and Hugging Face endpoints.
  - `3.Embededmodels/`: Embedding generation and vector similarity.
  - `parser/`: Prompt templates, chaining with `|`, and output parsers.
  - `promttemplate.py`: PromptTemplate and ChatPromptTemplate usage.
  - `promtting _technique/`: Prompting techniques, chat history, streamlit UI demo.
  - `structured_output/`: Structured outputs via JSON schema, Pydantic, TypedDict.
  - Top-level scripts: `1.py` (structured chat loop), `2.py` (HF image gen), `3.py` (OpenAI chat).

## Environment & Secrets
- `.env` is required; most scripts call `load_dotenv()`.
- Expected keys (set only those you need):
  - `OPENAI_API_KEY` (OpenAI + `langchain-openai`)
  - `GROQ_API_KEY` (Groq via `langchain-groq`)
  - `HF_TOKEN` (Hugging Face Hub / endpoints)
  - Possibly `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY` if you add those providers
- Dependencies are in `requirement.txt` (note singular filename).

## Install & Run
- Create venv (a `venv/` exists; reuse or create your own), then:
  - `pip install -r requirement.txt`
- Examples (zsh):
  - `python 2.chatModels/openai_demo.py`
  - `python 2.chatModels/groqapi_demo.py`
  - `python 2.chatModels/localOpen.py`
  - `python 2.chatModels/huggingface_demo.py`
  - `python 3.Embededmodels/opensouce_demo.py`
  - `python 3.Embededmodels/similarity_between_vectors.py`
  - `python parser/chain_parser.py`
  - `python parser/stroutputparser.py`
  - `python structured_output/using_pydantic.py`
  - Streamlit demo: `streamlit run "promtting _technique/dynamicPromtTEM.py"`

## Common Patterns
- Always `from dotenv import load_dotenv; load_dotenv()` before client/model init.
- Chat models:
  - OpenAI: `from langchain_openai import ChatOpenAI; ChatOpenAI(model="gpt-4.1-mini", temperature=0)`
  - Groq: `from langchain_groq import ChatGroq; ChatGroq(model="llama-3.3-70b-versatile", temperature=0)`
  - HF endpoint: `HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b", task="text-generation")` then `ChatHuggingFace(llm=llm)`
- Invocation:
  - Plain text: `result = llm.invoke("your prompt")`; access with `result.content` when using chat models.
  - Chat history: use `langchain.messages` (`HumanMessage`, `AIMessage`, `SystemMessage`) and pass a list to `.invoke(...)`.
- Prompting:
  - `PromptTemplate(...).format(...)` for strings; or `ChatPromptTemplate.from_messages([...]).invoke({...})` to produce a `PromptValue` or message list.
  - `MessagesPlaceholder(variable_name="chat_history")` lets you mix prior messages into a chat template.
- Chaining & parsing:
  - Compose with `|`: `template | model | StrOutputParser()`.
  - `JsonOutputParser()` is available; structured outputs below are preferred when schema is known.

## Structured Outputs
- JSON schema: `model.with_structured_output(schema_dict)` (see `structured_output/jsonstructure.py`).
- Pydantic: define model class and pass to `with_structured_output(PydanticModel)` (see `structured_output/using_pydantic.py`).
- TypedDict: pass a `TypedDict` for lightweight schemas (see `structured_output/usingfunction.py`).

## Embeddings & Similarity
- Embeddings: `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")`.
- Similarity: compute with `sklearn.metrics.pairwise.cosine_similarity` (see `3.Embededmodels/similarity_between_vectors.py`).

## Image Generation (HF Hub)
- See `2.py` using `huggingface_hub.InferenceClient(provider="nscale")` and `text_to_image(...)`.
- Requires a valid `HF_TOKEN` in `.env` for most providers/models.

## Conventions & Tips
- Scripts are independent; place new demos in the matching folder.
- Prefer `ChatGroq(model="llama-3.3-70b-versatile")` for Groq examples to stay consistent.
- Use relative paths for data files (e.g., `chat_history.txt`), not absolute system paths.
- Hugging Face chat models expect text or message inputs; use descriptive string prompts or `ChatPromptTemplate`.
- Folder name `promtting _technique/` contains a space; quote the path when needed in shell commands.

## Troubleshooting
- If `result` prints as an object, access `result.content` (chat models) or the returned dict (structured output).
- If a HF call fails, check `HF_TOKEN` and `repo_id`/`task` pair.
- If Streamlit isn’t found, `pip install streamlit` and run via Streamlit CLI.

