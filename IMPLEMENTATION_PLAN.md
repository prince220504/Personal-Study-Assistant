# Implementation Plan — Personal Study Assistant (RAG + LangGraph)

Derived from `study-assistant-rag-project-plan.md`. 2-day build, 1-day buffer. Teach-while-build.

---

## 1. Project Layout

```
Personal Study Assistant/
├── .env                      # GROQ_API_KEY
├── .gitignore
├── requirements.txt
├── data/
│   └── notes/                # user drops PDFs / .txt here
├── chroma_db/                # persisted vector store (auto-created)
├── src/
│   ├── __init__.py
│   ├── config.py             # env, paths, model names, constants
│   ├── ingest.py             # load → chunk → embed → persist
│   ├── rag_chain.py          # retriever + prompt + ChatGroq chain
│   ├── quiz_graph.py         # LangGraph: state, nodes, edges
│   └── prompts.py            # all prompt templates one place
├── app.py                    # Streamlit entrypoint (multipage)
├── pages/
│   ├── 1_Ask.py              # RAG Q&A page
│   └── 2_Quiz_Me.py          # Quiz agent page
└── IMPLEMENTATION_PLAN.md
```

Modular = swap pieces easy. Streamlit auto-detects `pages/` dir.

---

## 2. Dependencies

`requirements.txt`:
```
langchain
langchain-community
langchain-groq
langchain-huggingface
langgraph
chromadb
streamlit
pypdf
sentence-transformers
python-dotenv
```

Install: `pip install -r requirements.txt`

---

## 3. Environment

`.env`:
```
GROQ_API_KEY=gsk_xxx_from_console_groq_com
```

`.gitignore`:
```
.env
chroma_db/
__pycache__/
*.pyc
data/notes/*
!data/notes/.gitkeep
```

---

## 4. Per-File Responsibilities

### `src/config.py`
Centralize constants. Edit once, propagate everywhere.

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
NOTES_DIR = ROOT / "data" / "notes"
CHROMA_DIR = ROOT / "chroma_db"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.1-8b-instant"   # fast. swap to llama-3.3-70b-versatile for quality
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4
```

### `src/ingest.py`
One-time (or on-demand) pipeline: read files → split → embed → save.

**Concept — chunking**: LLM context limited. Split big docs into ~800-char windows w/ 100-char overlap. Overlap keeps sentences spanning boundaries retrievable.

**Concept — embeddings**: Each chunk → fixed-length vector. Semantically similar text → close vectors. Lets us search by meaning, not keywords.

**Concept — vector store**: DB indexed for vector similarity. Chroma = local SQLite-backed, no server.

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from .config import NOTES_DIR, CHROMA_DIR, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def load_docs():
    pdf_loader = DirectoryLoader(str(NOTES_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(str(NOTES_DIR), glob="**/*.txt", loader_cls=TextLoader)
    return pdf_loader.load() + txt_loader.load()

def build_vectorstore(force_rebuild=False):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if CHROMA_DIR.exists() and not force_rebuild:
        return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)

    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    vs = Chroma.from_documents(chunks, embeddings, persist_directory=str(CHROMA_DIR))
    return vs

if __name__ == "__main__":
    vs = build_vectorstore(force_rebuild=True)
    print(f"Indexed {vs._collection.count()} chunks.")
```

**Pitfall**: never re-embed every Streamlit run. Check `CHROMA_DIR` exists, reuse. Embedding model first-load downloads ~130MB — warn user.

### `src/prompts.py`
All prompt templates one file = easy tuning.

```python
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template("""
Answer the question using only the context below. If the context does not contain the answer, say so.

Context:
{context}

Question: {question}

Answer:
""")

QUIZ_GEN_PROMPT = ChatPromptTemplate.from_template("""
You are a quiz generator. From the study material below, write ONE clear question testing understanding (not trivia).
Return JSON: {{"question": "...", "expected_answer": "..."}}

Material:
{context}
""")

GRADE_PROMPT = ChatPromptTemplate.from_template("""
Grade the student's answer against the expected answer and source material.
Return JSON: {{"verdict": "correct"|"partial"|"incorrect", "feedback": "..."}}

Question: {question}
Expected: {expected}
Student answer: {answer}
Source: {context}
""")

HINT_PROMPT = ChatPromptTemplate.from_template("""
Student got the question wrong/partial. Give ONE short hint (no full answer).

Question: {question}
Their answer: {answer}
Source: {context}

Hint:
""")
```

### `src/rag_chain.py`
LangChain pipeline: retrieve → format → LLM → parse.

**Concept — retriever**: wraps vector store. `.invoke(query)` returns top-k similar chunks.

**Concept — chain (LCEL)**: `prompt | llm | parser` pipes data through stages.

```python
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .ingest import build_vectorstore
from .prompts import RAG_PROMPT
from .config import GROQ_API_KEY, LLM_MODEL, TOP_K

def format_docs(docs):
    return "\n\n".join(f"[Source: {d.metadata.get('source','?')} p.{d.metadata.get('page','?')}]\n{d.page_content}" for d in docs)

def build_rag_chain():
    vs = build_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever  # return retriever too for showing sources
```

### `src/quiz_graph.py`
LangGraph = state machine. Nodes mutate state. Edges route based on state.

**Concept — State**: typed dict shared across nodes.
**Concept — Node**: function `(state) -> dict_of_updates`.
**Concept — Edge**: connects nodes. Conditional edge = function that picks next node based on state.

```python
from typing import TypedDict, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
import json
from .ingest import build_vectorstore
from .prompts import QUIZ_GEN_PROMPT, GRADE_PROMPT, HINT_PROMPT
from .config import GROQ_API_KEY, LLM_MODEL, TOP_K

class QuizState(TypedDict):
    topic: str
    context: str
    question: Optional[str]
    expected: Optional[str]
    user_answer: Optional[str]
    verdict: Optional[str]      # correct | partial | incorrect
    feedback: Optional[str]
    hint: Optional[str]
    retries: int
    asked_count: int
    max_questions: int
    score: int
    weak_topics: List[str]
    history: List[dict]

llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.3)
vs = build_vectorstore()
retriever = vs.as_retriever(search_kwargs={"k": TOP_K})

def retrieve_topic_content(state: QuizState):
    docs = retriever.invoke(state["topic"])
    ctx = "\n\n".join(d.page_content for d in docs)
    return {"context": ctx}

def generate_question(state: QuizState):
    out = (QUIZ_GEN_PROMPT | llm).invoke({"context": state["context"]}).content
    data = json.loads(out)  # wrap in try/except in real code
    return {"question": data["question"], "expected": data["expected_answer"], "retries": 0}

def grade_answer(state: QuizState):
    out = (GRADE_PROMPT | llm).invoke({
        "question": state["question"], "expected": state["expected"],
        "answer": state["user_answer"], "context": state["context"],
    }).content
    data = json.loads(out)
    return {"verdict": data["verdict"], "feedback": data["feedback"]}

def give_hint(state: QuizState):
    out = (HINT_PROMPT | llm).invoke({
        "question": state["question"], "answer": state["user_answer"], "context": state["context"]
    }).content
    return {"hint": out, "retries": state["retries"] + 1}

def next_question(state: QuizState):
    correct = state["verdict"] == "correct"
    updates = {
        "asked_count": state["asked_count"] + 1,
        "score": state["score"] + (1 if correct else 0),
        "history": state["history"] + [{
            "q": state["question"], "verdict": state["verdict"], "topic": state["topic"]
        }],
        "weak_topics": state["weak_topics"] + ([state["topic"]] if not correct else []),
        "user_answer": None, "verdict": None, "hint": None, "feedback": None,
    }
    return updates

def route_after_grade(state: QuizState) -> Literal["hint", "next", "done"]:
    if state["verdict"] == "correct":
        return "next"
    if state["retries"] >= 1:    # 1 retry max
        return "next"
    return "hint"

def route_after_next(state: QuizState) -> Literal["new_q", "end"]:
    return "end" if state["asked_count"] >= state["max_questions"] else "new_q"

def build_quiz_graph():
    g = StateGraph(QuizState)
    g.add_node("retrieve", retrieve_topic_content)
    g.add_node("generate", generate_question)
    g.add_node("grade", grade_answer)
    g.add_node("hint", give_hint)
    g.add_node("advance", next_question)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    # generate → STOP (wait for user input via Streamlit); resume by invoking grade
    g.add_edge("generate", END)

    # second sub-graph or same graph w/ checkpointer used for grading turn
    g.add_conditional_edges("grade", route_after_grade, {"hint": "hint", "next": "advance", "done": END})
    g.add_edge("hint", END)        # wait for retry input
    g.add_conditional_edges("advance", route_after_next, {"new_q": "retrieve", "end": END})

    return g.compile()
```

**Note**: Streamlit + LangGraph interaction = tricky. Two clean options:
- **A (simple)**: invoke graph step-by-step from Streamlit; store `QuizState` in `st.session_state`; call individual node functions directly between user inputs.
- **B (proper)**: use LangGraph `MemorySaver` checkpointer + `interrupt_before=["grade"]`; persist thread_id in session_state.

Start with **A** for Day 2. If time, upgrade to B.

### `app.py`
Streamlit landing page. Sidebar for ingestion trigger.

```python
import streamlit as st
from pathlib import Path
from src.ingest import build_vectorstore
from src.config import NOTES_DIR, CHROMA_DIR

st.set_page_config(page_title="Study Assistant", page_icon="📚")
st.title("📚 Personal Study Assistant")
st.write("Drop PDFs/TXT into `data/notes/` then build the index.")

with st.sidebar:
    st.header("Index")
    st.write(f"Notes dir: `{NOTES_DIR}`")
    st.write(f"Index exists: {CHROMA_DIR.exists()}")
    if st.button("Rebuild index"):
        with st.spinner("Embedding..."):
            vs = build_vectorstore(force_rebuild=True)
        st.success(f"Indexed {vs._collection.count()} chunks.")

st.info("Use sidebar pages: **Ask** for Q&A, **Quiz Me** for self-test.")
```

### `pages/1_Ask.py`
```python
import streamlit as st
from src.rag_chain import build_rag_chain

st.title("Ask")

if "rag" not in st.session_state:
    st.session_state.rag, st.session_state.retriever = build_rag_chain()

q = st.text_input("Question:")
if q:
    with st.spinner("Thinking..."):
        ans = st.session_state.rag.invoke(q)
        sources = st.session_state.retriever.invoke(q)
    st.markdown("### Answer")
    st.write(ans)
    with st.expander("Sources"):
        for d in sources:
            st.markdown(f"**{d.metadata.get('source','?')}** (p.{d.metadata.get('page','?')})")
            st.write(d.page_content[:400] + "...")
```

### `pages/2_Quiz_Me.py`
State-machine in session_state. Pseudo:

```python
import streamlit as st
from src.quiz_graph import (
    retrieve_topic_content, generate_question, grade_answer,
    give_hint, next_question, QuizState
)

st.title("Quiz Me")

if "quiz" not in st.session_state:
    st.session_state.quiz = None

if st.session_state.quiz is None:
    topic = st.text_input("Topic")
    n = st.number_input("Questions", 1, 20, 5)
    if st.button("Start") and topic:
        state: QuizState = {
            "topic": topic, "context": "", "question": None, "expected": None,
            "user_answer": None, "verdict": None, "feedback": None, "hint": None,
            "retries": 0, "asked_count": 0, "max_questions": n,
            "score": 0, "weak_topics": [], "history": [],
        }
        state.update(retrieve_topic_content(state))
        state.update(generate_question(state))
        st.session_state.quiz = state
        st.rerun()
else:
    s = st.session_state.quiz
    if s["asked_count"] >= s["max_questions"]:
        st.success(f"Score: {s['score']}/{s['max_questions']}")
        st.write("Weak topics:", set(s["weak_topics"]) or "none")
        if st.button("Restart"):
            st.session_state.quiz = None; st.rerun()
    else:
        st.markdown(f"**Q{s['asked_count']+1}:** {s['question']}")
        if s.get("hint"):
            st.info(f"Hint: {s['hint']}")
        ans = st.text_area("Your answer", key=f"a_{s['asked_count']}_{s['retries']}")
        if st.button("Submit"):
            s["user_answer"] = ans
            s.update(grade_answer(s))
            st.write(f"Verdict: **{s['verdict']}** — {s['feedback']}")
            if s["verdict"] == "correct" or s["retries"] >= 1:
                s.update(next_question(s))
                if s["asked_count"] < s["max_questions"]:
                    s.update(retrieve_topic_content(s))
                    s.update(generate_question(s))
            else:
                s.update(give_hint(s))
            st.session_state.quiz = s
            st.rerun()
```

---

## 5. Build Order (Step-by-Step)

### Day 1 — RAG
1. Create folder layout, `.env`, `.gitignore`, `requirements.txt`. Install deps.
2. Drop 2-3 sample PDFs into `data/notes/`.
3. Write `config.py`, `prompts.py`.
4. Write `ingest.py`. Run `python -m src.ingest`. Verify `chroma_db/` created, chunk count > 0.
5. Write `rag_chain.py`. Quick test:
   ```python
   from src.rag_chain import build_rag_chain
   chain, _ = build_rag_chain()
   print(chain.invoke("what is X?"))
   ```
6. Write `app.py` + `pages/1_Ask.py`. Run `streamlit run app.py`. Ask question. Confirm answer + sources.

**Day 1 done when**: Streamlit Ask page returns grounded answer w/ source snippet.

### Day 2 — Quiz Graph
1. Write `quiz_graph.py`: state typeddict + node functions + (optional) compiled graph.
2. CLI smoke test the nodes outside Streamlit:
   ```python
   s = {...initial state...}
   s.update(retrieve_topic_content(s)); s.update(generate_question(s))
   print(s["question"], s["expected"])
   ```
3. Write `pages/2_Quiz_Me.py`. Run full quiz loop in browser.
4. Verify: question generates → submit wrong answer → hint appears → retry → next question → final score shows weak topics.

**Day 2 done when**: full quiz cycle ends w/ score + weak-topic list.

### Day 3 — Polish (optional)
- Better JSON parsing: use `langchain.output_parsers.JsonOutputParser` w/ Pydantic schema instead of raw `json.loads`.
- Show source citations in quiz mode (which PDF/page).
- "Re-quiz weak topics" button → set topic = random pick from `weak_topics`.
- Progress bar: `st.progress(asked_count / max_questions)`.
- Sidebar: topic dropdown built from filenames/metadata.
- Swap LLM to `llama-3.3-70b-versatile` for harder questions.

---

## 6. Concept Cheatsheet (teach-as-you-build)

| Term | Plain meaning |
|------|---------------|
| Chunk | Slice of doc small enough for LLM context |
| Overlap | Repeat chars between chunks so split sentences still findable |
| Embedding | Vector representation of text; similar meaning → close vectors |
| Vector store | DB indexed for nearest-vector lookup (Chroma) |
| Retriever | Wrapper that takes query string → returns top-k chunks |
| Chain (LCEL) | `prompt | llm | parser` pipeline; data flows left → right |
| Prompt template | String w/ `{vars}` filled at call time |
| **LangGraph state** | Shared dict mutated by nodes |
| **Node** | Function `(state) → partial state update` |
| **Edge** | Static link between nodes |
| **Conditional edge** | Function picks next node based on state |
| **Checkpointer** | Saves state per `thread_id` so graph can pause/resume |
| END | Sentinel marking graph exit |

LangChain = straight pipe. LangGraph = state machine w/ branches/loops (needed for quiz retries + topic looping).

---

## 7. Beginner Pitfalls (avoid before they bite)

| # | Trap | Fix |
|---|------|-----|
| 1 | Re-embedding every Streamlit run | Cache `build_vectorstore()` w/ `@st.cache_resource` or check `CHROMA_DIR` exists |
| 2 | First HF embedding load = slow download | Warn user; pre-run `python -m src.ingest` once |
| 3 | LLM returns non-JSON; `json.loads` crashes | Wrap try/except; use `JsonOutputParser` w/ retry |
| 4 | Streamlit reruns whole script each interaction | Persist state in `st.session_state`; cache chains w/ `@st.cache_resource` |
| 5 | Groq rate limit (free tier ~30 req/min) | Don't loop calls; show error gracefully |
| 6 | Chunks too big → cut off; too small → no context | 800/100 starting point; tune if answers shallow |
| 7 | LangGraph + Streamlit two-way streaming = hard | Use node-by-node imperative invocation (option A above) instead of graph w/ interrupts |
| 8 | `Chroma.from_documents` requires non-empty list | Guard against empty `data/notes/` |
| 9 | Forgetting `persist_directory` → DB lost on exit | Always pass `persist_directory=str(CHROMA_DIR)` |
| 10 | Mixing chunk source metadata loss | Loaders attach `source`/`page` automatically — display them |

---

## 8. Validation Per Day

**Day 1 acceptance**:
- [ ] `python -m src.ingest` finishes, prints chunk count
- [ ] `streamlit run app.py` opens UI
- [ ] Ask page answers a question from notes
- [ ] Source section shows ≥1 chunk w/ filename

**Day 2 acceptance**:
- [ ] Quiz page generates question from topic
- [ ] Wrong answer → hint shown, retry allowed
- [ ] Correct or retries used → next question loads
- [ ] After N questions: score + weak topics summary

**Day 3 acceptance** (if attempted):
- [ ] JSON parsing never crashes app
- [ ] Quiz shows source citation
- [ ] Re-quiz weak topics works

---

## 9. Git Workflow (user runs commands, AI guides)

Commit at end of each day. User executes; AI provides exact commands. Repo lives on GitHub.

### One-time setup (before Day 1 coding)

```bash
cd "C:\Users\Prince\OneDrive\Desktop\Personal Study Assistant"
git init
git branch -M main
```

`.gitignore` already excludes `.env`, `chroma_db/`, `__pycache__/`, `data/notes/*`. Verify before first commit:
```bash
git status
```
If `.env` shows as untracked — STOP, fix `.gitignore` first. Never commit API keys.

Create empty GitHub repo at https://github.com/new (name: `personal-study-assistant`, private recommended). Do NOT initialize w/ README/license/.gitignore (we have local files already).

Connect:
```bash
git remote add origin https://github.com/<your-username>/personal-study-assistant.git
```

### Day 1 commit (end of Day 1)

After Ask page works:
```bash
git status                 # review what's being added
git add .
git status                 # confirm .env NOT staged
git commit -m "Day 1: RAG pipeline + Streamlit Ask page

- Ingest PDFs/TXT from data/notes via DirectoryLoader
- Chunk w/ RecursiveCharacterTextSplitter (800/100)
- Embed w/ BAAI/bge-small-en-v1.5, persist to Chroma
- RAG chain: retriever | prompt | ChatGroq | StrOutputParser
- Streamlit Ask page returns answer + source citations"
git push -u origin main
```

`-u origin main` sets upstream — future pushes need only `git push`.

### Day 2 commit (end of Day 2)

After Quiz page works:
```bash
git status
git add .
git commit -m "Day 2: LangGraph quiz agent + Streamlit Quiz page

- QuizState TypedDict: topic, question, expected, verdict, retries, score, weak_topics
- Nodes: retrieve_topic_content, generate_question, grade_answer, give_hint, next_question
- Conditional edges: route_after_grade (hint vs next), route_after_next (new_q vs end)
- Streamlit page: imperative node invocation w/ st.session_state (option A)
- Final summary: score + weak topics list"
git push
```

### Day 3 commit (if polish done)

```bash
git add .
git commit -m "Day 3: polish — JSON parser, source citations in quiz, progress bar"
git push
```

### Mid-day safety commits (optional but recommended)

If finishing a working sub-step (e.g., ingest works but RAG chain not done yet):
```bash
git add src/config.py src/ingest.py requirements.txt .gitignore
git commit -m "WIP: ingest pipeline works, chunk count verified"
```
Smaller commits = easier to roll back if something breaks later.

### Pitfalls

| # | Trap | Fix |
|---|------|-----|
| 1 | `.env` accidentally committed | Check `git status` BEFORE `git commit`. If leaked → rotate Groq key immediately + `git rm --cached .env` |
| 2 | `chroma_db/` (100MB+) committed | Confirm `.gitignore` lists it before first `git add .` |
| 3 | Sample PDFs committed (copyright issue) | `data/notes/*` already in `.gitignore`, only `.gitkeep` tracked |
| 4 | Push rejected (remote has commits) | Should not happen if remote created empty. If it does: `git pull --rebase origin main` then push |
| 5 | Wrong author identity | One-time: `git config --global user.name "Your Name"` + `git config --global user.email "you@example.com"` |
| 6 | Forgot `-u` on first push | Use `git push origin main` until set, or rerun `git push -u origin main` |

### Checklist before each commit

- [ ] `git status` shows no `.env`, no `chroma_db/`, no large data files
- [ ] App actually runs (`streamlit run app.py` still works)
- [ ] Commit message describes WHAT changed + WHY (template above)
- [ ] After push: refresh GitHub page, confirm files visible

---

## 10. Stretch Ideas (post-MVP, not in 3 days)

- Persistent quiz history (SQLite)
- Spaced repetition scheduler
- Multi-doc summarization page
- Streamlit auth / multi-user
- Swap Chroma → FAISS for speed
- Stream LLM tokens to UI

---

## 11. Reference Links

- Groq console: https://console.groq.com
- LangChain docs: https://python.langchain.com
- LangGraph docs: https://langchain-ai.github.io/langgraph/
- Chroma: https://docs.trychroma.com
- bge-small-en: https://huggingface.co/BAAI/bge-small-en-v1.5
