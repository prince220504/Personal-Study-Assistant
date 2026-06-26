<div align="center">

# 📚 Personal Study Assistant

### Your AI-Powered Study Buddy — Ask Questions & Take Quizzes from Your Own PDFs

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://python.langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0A192F?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge)](https://groq.com)

---

*Upload your PDF notes → Ask grounded questions with citations → Take AI-generated quizzes with hints, retries & weak-topic tracking*

</div>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 📖 Ask — RAG Q&A
- Upload PDFs and ask anything
- Answers grounded **only** in your notes
- Source citations with file name & page number
- Structured responses: definition → simple explanation → key points → example

</td>
<td width="50%">

### 🧠 Quiz Me — LangGraph Agent
- Pick a topic from your PDFs or type your own
- AI generates questions from your actual notes
- Wrong answer? Get a **hint** and retry once
- Final score + **weak topics** tracking
- **Re-quiz weak topics** with one click

</td>
</tr>
</table>

### 🔥 Highlights

| Feature | Description |
|:--------|:------------|
| 🎯 **Source Citations** | Every answer and quiz question shows exactly which PDF and page it came from |
| 🔄 **Smart Retry Flow** | Wrong answer → hint → one retry → advance. LangGraph handles the state machine |
| 📊 **Weak Topic Tracking** | See which topics you struggled with + re-quiz them instantly |
| 📂 **Topic Dropdown** | Auto-populated from your uploaded PDF filenames — no guessing what to quiz on |
| 🛡️ **Robust JSON Parsing** | Pydantic schemas + fallback parser — the quiz never crashes from LLM output issues |
| ⚡ **Zero Cost** | Groq free tier + local HuggingFace embeddings + local Chroma DB |

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|:------|:-----------|:--------|
| 🖥️ **UI** | Streamlit | Multipage web interface |
| 🔗 **RAG** | LangChain (LCEL) | Retrieve → Prompt → LLM → Answer pipeline |
| 🧩 **Quiz Agent** | LangGraph | Stateful graph with branches, loops & checkpointing |
| 🤖 **LLM** | Groq — Llama 3.3 70B | Fast, free-tier inference |
| 📐 **Embeddings** | HuggingFace `bge-small-en-v1.5` | Local, free, ~130MB |
| 🗄️ **Vector DB** | Chroma (local) | File-backed, no server setup |
| 📄 **PDF Parsing** | pypdf | Via LangChain document loaders |

</div>

---

## 📁 Project Structure

```
📚 Personal Study Assistant/
│
├── 🏠 app.py                    # Streamlit home page (entry point)
├── 📄 pages/
│   ├── 1_Ask.py                 # RAG Q&A page
│   └── 2_Quiz_Me.py             # LangGraph quiz page
│
├── ⚙️ src/
│   ├── config.py                # Paths, model names, API keys
│   ├── ingest.py                # Load → Chunk → Embed → Store
│   ├── rag_chain.py             # LangChain RAG pipeline
│   ├── quiz_graph.py            # LangGraph state machine
│   ├── prompts.py               # All prompt templates
│   ├── schemas.py               # Pydantic output schemas
│   └── upload_ui.py             # Shared PDF upload sidebar
│
├── 📂 data/uploads/             # User-uploaded PDFs (gitignored)
├── 🗄️ user_chroma_db/           # Vector index (auto-created, gitignored)
│
├── 📋 IMPLEMENTATION_PLAN.md    # Original build plan
└── 📝 requirements.txt          # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### Installation

**1. Clone the repo**
```bash
git clone https://github.com/prince220504/Personal-Study-Assistant.git
cd Personal-Study-Assistant
```

**2. Create virtual environment & install dependencies**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

**3. Set up environment variables**

Copy the example and fill in your key:
```powershell
copy .env.example .env
```

Then edit `.env`:
```env
GROQ_API_KEY=gsk_your_actual_key_here
HF_TOKEN=hf_your_token_here          # optional — reduces download warnings
```

**4. Launch the app**
```powershell
streamlit run app.py
```

---

## 📖 How To Use

```
Step 1  →  Open the app in your browser
Step 2  →  Upload PDF notes from the sidebar
Step 3  →  Click "Save uploaded PDFs"
Step 4  →  Click "Build index from my PDFs"
Step 5  →  Use Ask to ask questions with source citations
Step 6  →  Use Quiz Me — pick a topic from the dropdown or type your own
Step 7  →  After the quiz, hit "Re-quiz weak topics" to drill your weak spots
```

---

## 🔒 Privacy

> Your data stays local. PDFs are stored in `data/uploads/`, and the vector index lives in `user_chroma_db/`. Both are gitignored. Only the question text and retrieved chunks are sent to the Groq API — never your full PDFs.

---

## 📊 Architecture

```
┌──────────────────────────────────────────────────────┐
│                    STREAMLIT UI                       │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │   Home   │  │   Ask    │  │     Quiz Me        │  │
│  │          │  │  (RAG)   │  │   (LangGraph)      │  │
│  └──────────┘  └────┬─────┘  └────────┬───────────┘  │
│                     │                 │               │
└─────────────────────┼─────────────────┼───────────────┘
                      │                 │
              ┌───────▼───────┐  ┌──────▼──────────┐
              │  rag_chain.py │  │  quiz_graph.py   │
              │  (LCEL pipe)  │  │  (StateGraph)    │
              └───────┬───────┘  └──────┬──────────┘
                      │                 │
              ┌───────▼─────────────────▼──────────┐
              │           ingest.py                 │
              │    Load → Chunk → Embed → Store     │
              └───────────────┬────────────────────┘
                              │
                 ┌────────────▼────────────┐
                 │    Chroma Vector DB     │
                 │   (user_chroma_db/)     │
                 └─────────────────────────┘
```

---

## 🗺️ Build Journey

| Step | Focus | What Got Built |
|:---:|:------|:---------------|
| **1** | **Core RAG Pipeline** | PDF/TXT ingestion → chunking → embeddings → Chroma vector store → LCEL retrieval chain → Streamlit Ask page with source citations |
| **2** | **LangGraph Quiz Agent** | `QuizState` TypedDict → 5 nodes (retrieve, generate, grade, hint, advance) → conditional routing → compiled `StateGraph` with `MemorySaver` checkpointer → Streamlit Quiz page with full quiz loop |
| **3** | **Polish** | In-app PDF upload widget · Pydantic schemas + robust JSON parsing · Source citations in quiz · Re-quiz weak topics button · Topic dropdown from PDF names |

---

<div align="center">

**Built with ❤️ while learning LangChain & LangGraph**

*Star ⭐ this repo if you found it useful!*

</div>
