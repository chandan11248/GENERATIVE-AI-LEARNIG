# 🦜🔗 LangChain Mastery - Generative AI Learning Journey

A comprehensive repository documenting my journey learning **LangChain**, **RAG**, **Agents**, and **Generative AI** concepts with hands-on projects.

---

## 🎯 Overview

This repository contains practical implementations, notebooks, and projects covering the complete LangChain ecosystem - from basic chat models to advanced agentic workflows.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     🦜 LANGCHAIN LEARNING ROADMAP                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│   FOUNDATION  │           │   ADVANCED    │           │   PROJECTS    │
│               │           │               │           │               │
│ • Chat Models │           │ • RAG         │           │ • YouTube Bot │
│ • Prompts     │           │ • Agents      │           │ • Web Search  │
│ • Parsers     │           │ • Tools       │           │               │
│ • Chains      │           │ • Vector DBs  │           │               │
└───────────────┘           └───────────────┘           └───────────────┘
```

---

## 📂 Repository Structure

```
LangChain/
│
├── 📁 2.chatModels/              # LLM Integration
│   ├── groqapi_demo.py           # Groq API usage
│   ├── huggingface_demo.py       # HuggingFace models
│   ├── openai_demo.py            # OpenAI integration
│   └── localOpen.py              # Local LLM setup
│
├── 📁 prompting_technique/       # Prompt Engineering
│   ├── simplechatbot.py          # Basic chatbot
│   ├── dynamicPromtTEM.py        # Dynamic templates
│   ├── messageplaceholder.py     # Message placeholders
│   └── qdynamicpromtingchatbot.py# Advanced chatbot
│
├── 📁 parser/                    # Output Parsers
│   ├── stroutputparser.py        # String parser
│   ├── jsonparser.py             # JSON parser
│   ├── pydanticoutputparser.py   # Pydantic parser
│   └── chain_parser.py           # Chain with parser
│
├── 📁 chains/                    # LangChain Chains
│   ├── sequential_chain.ipynb    # Sequential execution
│   ├── runnable_parallel.ipynb   # Parallel execution
│   ├── runnable_passthrough.ipynb# Passthrough runnable
│   ├── runnable_lambda.ipynb     # Lambda functions
│   ├── branch_runnable.ipynb     # Branching logic
│   └── conditional_chain.ipynb   # Conditional flows
│
├── 📁 structured_output/         # Structured Outputs
│   ├── using_pydantic.py         # Pydantic models
│   ├── jsonstructure.py          # JSON structure
│   └── usingfunction.py          # Function calling
│
├── 📁 document_retriever/        # Document Loading
│   ├── text_loader.ipynb         # Text file loading
│   └── pdfs/                     # PDF documents
│
├── 📁 text-splitter/             # Text Splitting
│   └── splitter.ipynb            # Various splitters
│
├── 📁 vector database/           # Vector Stores
│   ├── vectordb.ipynb            # ChromaDB basics
│   ├── youtube_chatbot.ipynb     # YouTube RAG
│   ├── my_chroma_db/             # Persistent DB
│   └── sample_db/                # Sample database
│
├── 📁 tools/                     # LangChain Tools
│   ├── tools_in_LC.ipynb         # Tool creation
│   └── agents_in_lang_chain.ipynb# ReAct agents
│
├── 📁 projects/                  # Complete Projects
│   ├── youtube_chatbot/          # YouTube Video Bot
│   │   └── youtube_bot.py        # Streamlit app
│   └── web_search_using_openLLM.ipynb # Hybrid RAG
│
└── 📄 README.md                  # This file
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Framework** | LangChain, LangGraph |
| **LLMs** | Groq (Llama 3), OpenAI, HuggingFace |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector Store** | ChromaDB, FAISS |
| **Web Search** | Tavily, DuckDuckGo |
| **UI** | Streamlit |
| **APIs** | YouTube Transcript, WeatherStack |

---

## 📚 Learning Modules

### Module 1: Chat Models 💬

Learn to integrate various LLM providers:

```python
# Groq Example
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama3-8b-8192")

# OpenAI Example
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")

# HuggingFace Example
from langchain_huggingface import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-2-7b-chat-hf")
```

**Files:** `2.chatModels/`

---

### Module 2: Prompt Engineering 📝

Master prompt templates and techniques:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
```

**Files:** `prompting_technique/`

---

### Module 3: Output Parsers 🔄

Structure LLM outputs:

```
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT PARSERS                            │
├─────────────────────────────────────────────────────────────┤
│  StrOutputParser    → Plain text output                     │
│  JsonOutputParser   → JSON structured output                │
│  PydanticParser     → Validated Python objects              │
└─────────────────────────────────────────────────────────────┘
```

**Files:** `parser/`

---

### Module 4: Chains ⛓️

Build complex workflows:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          CHAIN TYPES                                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Sequential Chain        Parallel Chain         Conditional Chain        │
│  ┌───┐ ┌───┐ ┌───┐      ┌───┐                  ┌───┐                     │
│  │ A │→│ B │→│ C │      │ A │                  │ A │                     │
│  └───┘ └───┘ └───┘      └─┬─┘                  └─┬─┘                     │
│                       ┌───┴───┐               ┌──┴──┐                    │
│                       ▼       ▼               ▼     ▼                    │
│                     ┌───┐   ┌───┐           ┌───┐ ┌───┐                  │
│                     │ B │   │ C │           │ B │ │ C │                  │
│                     └───┘   └───┘           └───┘ └───┘                  │
│                       │       │             (if)  (else)                 │
│                       └───┬───┘                                          │
│                           ▼                                              │
│                         ┌───┐                                            │
│                         │ D │                                            │
│                         └───┘                                            │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

**Files:** `chains/`

---

### Module 5: Document Processing 📄

Load and split documents:

```python
# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
```

**Files:** `document_retriever/`, `text-splitter/`

---

### Module 6: Vector Databases 🗄️

Store and retrieve embeddings:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VECTOR DATABASE WORKFLOW                          │
└─────────────────────────────────────────────────────────────────────────┘

   Documents          Chunks           Embeddings         Vector Store
  ┌─────────┐       ┌───────┐        ┌───────────┐       ┌───────────┐
  │  PDF    │       │ Chunk │        │ [0.1, 0.2 │       │           │
  │  TXT    │  ───► │ Chunk │  ───►  │  0.3, ...]│  ───► │  ChromaDB │
  │  DOCX   │       │ Chunk │        │           │       │           │
  └─────────┘       └───────┘        └───────────┘       └───────────┘
                                                                │
                                                                ▼
                                                          ┌───────────┐
                                                          │  Query    │
                                                          │  Results  │
                                                          └───────────┘
```

**Files:** `vector database/`

---

### Module 7: Tools & Agents 🤖

Build intelligent agents:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ReAct AGENT LOOP                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     │
│    │ THOUGHT  │ ──► │  ACTION  │ ──► │   TOOL   │ ──► │OBSERVATION│     │
│    └──────────┘     └──────────┘     └──────────┘     └──────────┘     │
│         ▲                                                   │          │
│         └───────────────────────────────────────────────────┘          │
│                          (Loop until done)                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Files:** `tools/`

---

## 🚀 Projects

### Project 1: YouTube Video Chatbot 🎬

A RAG-based chatbot that answers questions from YouTube videos.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ YouTube URL  │ ──► │  Transcript  │ ──► │   Chunks     │
└──────────────┘     └──────────────┘     └──────────────┘
                            │                    │
                            ▼                    ▼
                     ┌──────────────┐     ┌──────────────┐
                     │  Translate   │     │  ChromaDB    │
                     │  (25+ langs) │     │  Embeddings  │
                     └──────────────┘     └──────────────┘
                                                │
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Answer     │ ◄── │   Groq LLM   │ ◄── │  Retriever   │
└──────────────┘     └──────────────┘     └──────────────┘
```

**Features:**
- 📹 YouTube transcript extraction
- 🌐 25+ language support with auto-translation
- 📝 AI-powered summaries
- 💬 Interactive Q&A chat

**Location:** `projects/youtube_chatbot/`

---

### Project 2: Hybrid RAG with Web Search 🔍

A hybrid system combining local documents with real-time web search.

```
┌───────────────────────────┐       ┌───────────────────────────┐
│   📚 LOCAL RETRIEVAL      │       │   🌐 WEB SEARCH           │
│   (ChromaDB)              │       │   (Tavily API)            │
└─────────────┬─────────────┘       └─────────────┬─────────────┘
              │                                   │
              └─────────────┬─────────────────────┘
                            ▼
                  ┌─────────────────┐
                  │ Combined Context│
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │   Groq LLM      │
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │   📝 Answer     │
                  └─────────────────┘
```

**Location:** `projects/web_search_using_openLLM.ipynb`

---

## ⚡ Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/LangChain.git
cd LangChain
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install langchain langchain-community langchain-groq langchain-huggingface
pip install langchain-chroma chromadb sentence-transformers
pip install streamlit youtube-transcript-api googletrans==4.0.0-rc1
pip install tavily-python duckduckgo-search
pip install python-dotenv
```

### 4. Setup Environment Variables

```bash
# Create .env file
echo "GROQ_API_KEY=your_groq_key" >> .env
echo "OPENAI_API_KEY=your_openai_key" >> .env
echo "TAVILY_API_KEY=your_tavily_key" >> .env
echo "HUGGINGFACEHUB_API_TOKEN=your_hf_token" >> .env
```

### 5. Run Projects

```bash
# YouTube Chatbot
cd projects/youtube_chatbot
streamlit run youtube_bot.py

# Or run notebooks
jupyter notebook
```

---

## 📊 Learning Path

```
Week 1: Foundations
├── Day 1-2: Chat Models (OpenAI, Groq, HuggingFace)
├── Day 3-4: Prompt Templates & Engineering
└── Day 5-7: Output Parsers

Week 2: Chains & Documents
├── Day 1-3: Chain Types (Sequential, Parallel, Conditional)
├── Day 4-5: Document Loaders & Text Splitters
└── Day 6-7: Structured Outputs

Week 3: RAG
├── Day 1-2: Embeddings & Vector Stores
├── Day 3-4: Retrieval Strategies
└── Day 5-7: Complete RAG Pipeline

Week 4: Agents & Projects
├── Day 1-2: Tools & Custom Tools
├── Day 3-4: ReAct Agents
└── Day 5-7: Build Complete Projects
```

---

## 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| **LLM** | Large Language Model - AI model for text generation |
| **RAG** | Retrieval-Augmented Generation - Combine retrieval with generation |
| **Embeddings** | Vector representations of text |
| **Vector Store** | Database for storing and searching embeddings |
| **Chain** | Sequence of operations in LangChain |
| **Agent** | AI that can use tools and make decisions |
| **Tool** | Function that an agent can use |
| **Prompt Template** | Reusable prompt structure |

---

## ⚠️ Important Notes

### LangChain Agents Limitations

```
┌─────────────────────────────────────────────────────────────────┐
│  LangChain Agents = Good for Learning, Not for Production       │
├─────────────────────────────────────────────────────────────────┤
│  ❌ Parsing errors                                               │
│  ❌ Infinite loops                                               │
│  ❌ Unpredictable behavior                                       │
│  ❌ Hard to debug                                                │
├─────────────────────────────────────────────────────────────────┤
│  ✅ Use LangGraph for production agentic workflows              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📚 Resources

| Resource | Link |
|----------|------|
| LangChain Docs | https://python.langchain.com/ |
| LangGraph Docs | https://langchain-ai.github.io/langgraph/ |
| Groq Console | https://console.groq.com/ |
| ChromaDB Docs | https://docs.trychroma.com/ |
| HuggingFace | https://huggingface.co/ |

---

## 🤝 Contributing

Feel free to:
- ⭐ Star this repository
- 🐛 Report issues
- 🔀 Submit pull requests
- 💡 Suggest improvements

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Chandan kumar shah**

- GitHub: [@chandan11248](https://github.com/chandan11248)
- Gmail: [letschandansah@gmail.com]()

---

<p align="center">
  <b>⭐ If you found this helpful, please star the repository! ⭐</b>
</p>

<p align="center">
  Made with ❤️ while learning Generative AI
</p>