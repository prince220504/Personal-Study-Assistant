# Project: Personal Study/Exam Assistant (RAG + LangChain + LangGraph)

## Goal
Build a study assistant that:
1. Answers questions from my own notes/PDFs using RAG (LangChain).
2. Has a "Quiz Me" mode that generates questions from my notes, grades my answers, gives hints, and tracks weak topics — built as a stateful agent using LangGraph.

I am learning LangChain and LangGraph WHILE building this. I know Python but am new to both frameworks. Explain key concepts briefly as you introduce them (chunking, embeddings, retrievers, LangGraph nodes/edges/state) — don't just dump code, help me understand what each piece does and why.

Timeline: 2 days, max 3. Keep scope tight. Working app > extra features.

---

## Tech Stack (free/cloud, no local LLM)
- **LLM**: Groq API (`langchain-groq`, model: `llama-3.1-8b-instant` or `llama-3.3-70b-versatile`)
- **Embeddings**: Local/free via HuggingFace `sentence-transformers` (`BAAI/bge-small-en-v1.5`) — no API cost
- **Vector store**: Chroma (local, file-based, no server setup)
- **Framework**: LangChain (RAG pipeline) + LangGraph (quiz agent with state/loops)
- **UI**: Streamlit
- **Docs**: My own notes/PDFs (will provide a folder of files)

### Required packages
```bash
pip install langchain langchain-community langchain-groq langgraph chromadb streamlit pypdf sentence-transformers langchain-huggingface
```

### Env
- `GROQ_API_KEY` set in `.env` (free key from https://console.groq.com)

---

## Day 1 — Core RAG Pipeline (LangChain)

**Steps:**
1. **Ingest**: Load PDFs/text notes from a local folder (`DirectoryLoader` + `PyPDFLoader`/`TextLoader`).
2. **Chunk**: `RecursiveCharacterTextSplitter`, chunk_size ~800, overlap ~100.
3. **Embed + store**: Embed chunks with `HuggingFaceEmbeddings` (`bge-small-en-v1.5`), store in Chroma (persist to disk so I don't re-embed every run).
4. **Retrieve + generate**: Build a retrieval chain — given a question, fetch top-k chunks, stuff into a prompt template, send to Groq LLM (`ChatGroq`), return answer + show which source chunk/doc it came from (basic citation).
5. **Streamlit page 1 ("Ask")**: Text input → question → display answer + source snippet.

**Deliverable for Day 1**: I can ask any question about my notes in the Streamlit app and get an answer with a source reference.

---

## Day 2 — LangGraph Quiz Agent

Build a graph (not a plain chain) because this flow has state and loops:

**State to track**: current topic, questions asked, score, weak topics list, current question, current expected answer/context.

**Nodes:**
- `retrieve_topic_content` — pull relevant chunks for a chosen topic (reuse Day 1 retriever)
- `generate_question` — LLM creates a question from those chunks (store expected-answer context for grading)
- `get_user_answer` — Streamlit text input feeds the user's answer into graph state
- `grade_answer` — LLM compares user's answer against source content, returns a score/verdict (correct / partially correct / incorrect)
- `give_hint_and_retry` — if incorrect/partial, generate a hint, let user retry (limit to 1-2 retries)
- `next_question` — if correct or retries exhausted, log result (update weak topics if wrong), move to next question

**Conditional edges**: based on grading result, route to hint/retry vs next question vs end-of-quiz summary.

**Streamlit page 2 ("Quiz Me")**: Pick a topic → start quiz → see question → submit answer → see feedback/hint → next question → final summary screen (score, weak topics).

**Deliverable for Day 2**: A working quiz loop with state, branching, and a final score/weak-topics summary.

---

## Day 3 (buffer/optional polish)
- Improve prompt templates for better question quality and grading accuracy.
- Add simple source citations in quiz mode too (show which doc/section a question came from).
- Polish Streamlit UI (sidebar for topic selection, progress bar, etc.)
- (Optional) Add a "weak topics" re-quiz button that pulls only from previously-missed topics.

---

## Teaching/Learning Instructions for the Coding AI
- As we build, briefly explain *why* each LangChain/LangGraph component is used before/while writing it (e.g., "we use a retriever here because...", "this is a LangGraph node, meaning...").
- Point out the conceptual difference in practice: LangChain = linear pipeline (retrieve → prompt → generate), LangGraph = stateful graph with branching/loops (used for the quiz logic).
- Keep code modular: separate files for ingestion, RAG chain, LangGraph quiz graph, and the Streamlit app — not one giant script.
- Flag any place where I'm likely to hit a common beginner mistake (e.g., re-embedding on every run, prompt size limits, Groq rate limits) before I run into it.
