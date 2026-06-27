"""
Quiz Graph — LangGraph state machine for the Quiz Me feature.

The Streamlit page drives the compiled StateGraph. LangGraph keeps the quiz
state in a SQLite checkpointer (so in-progress quizzes survive an app
restart) and pauses before the grade node so Streamlit can collect the
user's answer.

Concepts used:
- State: shared dict every node reads/writes
- Node: function (state) -> partial state update
- Conditional routing: function (state) -> next-node name
- Skip-signal: page sets retries=1 to force the router to "advance" instead
  of "hint" (user chose to skip the question)
"""

import json
import re
from pathlib import Path
from typing import TypedDict, List, Optional, Literal

from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.exceptions import OutputParserException

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from .ingest import build_vectorstore
from .prompts import QUIZ_GEN_PROMPT, GRADE_PROMPT, HINT_PROMPT
from .schemas import QuizQuestion, GradeResult
from .config import GROQ_API_KEY, LLM_MODEL, TOP_K, ACTIVE_CHROMA_DIR


# ---------- State ----------

class QuizState(TypedDict):
    """
    The 'backpack' the quiz agent carries through the flow.
    Every node can read from it and return a partial update.
    """
    topic: str                       # what the user wants to be quizzed on
    context: str                     # retrieved chunks joined as one string
    source_citations: List[str]      # source file + page for each retrieved chunk
    question: Optional[str]          # current question text
    expected: Optional[str]          # expected answer (used for grading)
    user_answer: Optional[str]       # student's submitted answer
    verdict: Optional[str]           # "correct" | "partial" | "incorrect"
    feedback: Optional[str]          # short feedback from grader
    hint: Optional[str]              # hint shown after a wrong answer
    retries: int                     # retries used on current question (max 1)
    asked_count: int                 # how many questions answered so far
    max_questions: int               # total questions to ask
    score: int                       # number of fully correct answers
    weak_topics: List[str]           # topics where the user got it wrong
    history: List[dict]              # log of past questions and verdicts
    awaiting_continue: bool          # waiting for the user to click "Continue" before advancing


# ---------- Shared resources ----------
# The LLM can be built at import time, but the vector store cannot.
# Users upload PDFs during the Streamlit session, so retrieval is loaded
# lazily the first time a quiz asks for context.

_llm = ChatGroq(
    model=LLM_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.3,
)
_retriever = None


def reset_quiz_resources():
    """Clear cached retrieval resources after the user rebuilds the PDF index."""
    global _retriever
    _retriever = None


def _get_retriever():
    """Build/load the active user PDF retriever on demand."""
    global _retriever
    if _retriever is None:
        vs = build_vectorstore()
        _retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    return _retriever

# ---------- Output parsers ----------
# JsonOutputParser uses the Pydantic models to (a) generate format
# instructions injected into the prompt and (b) validate the LLM output.
# We cache the format-instruction strings here because they never change
# at runtime — building them once per import saves a few ms per call.

_question_parser = JsonOutputParser(pydantic_object=QuizQuestion)
_grade_parser = JsonOutputParser(pydantic_object=GradeResult)
_str_parser = StrOutputParser()

_question_format = _question_parser.get_format_instructions()
_grade_format = _grade_parser.get_format_instructions()

# Pre-built LCEL chains: prompt | llm | parser
# The parser hands us a validated dict (NOT a Pydantic instance — that's
# JsonOutputParser's behaviour; PydanticOutputParser would return the
# model instance instead, but a dict is simpler to merge into state).
_question_chain = QUIZ_GEN_PROMPT | _llm | _question_parser
_grade_chain = GRADE_PROMPT | _llm | _grade_parser
_hint_chain = HINT_PROMPT | _llm | _str_parser


# ---------- Helpers ----------

def _safe_json_loads(raw: str) -> dict:
    """
    LLMs sometimes wrap JSON in markdown fences or add stray text.
    Strip fences and try to extract the first {...} block before parsing.
    Raises ValueError with the raw output if parsing still fails.
    """
    cleaned = raw.strip()
    # Remove ```json ... ``` or ``` ... ``` fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: grab first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM output:\n{raw}")


def make_initial_state(topic: str, max_questions: int) -> QuizState:
    """Build a fresh QuizState dict to drop into st.session_state."""
    return {
        "topic": topic,
        "context": "",
        "source_citations": [],
        "question": None,
        "expected": None,
        "user_answer": None,
        "verdict": None,
        "feedback": None,
        "hint": None,
        "retries": 0,
        "asked_count": 0,
        "max_questions": max_questions,
        "score": 0,
        "weak_topics": [],
        "history": [],
        "awaiting_continue": False,
    }


# ---------- Nodes ----------

def retrieve_topic_content(state: QuizState) -> dict:
    """
    Node: pull top-k chunks from Chroma for the topic.
    Joins them into a single string the LLM can read.
    Also extracts source citations (filename + page) for UI display.

    For questions after the first, we perturb the retrieval query with the
    question number. Chroma's embedding similarity search then surfaces
    different chunks on each call instead of returning the same top-k every
    time. Combined with the history-aware QUIZ_GEN_PROMPT, this prevents
    Q2/Q3 from being near-duplicates of Q1.
    """
    asked = state.get("asked_count", 0)
    base_topic = state["topic"]
    query = base_topic if asked == 0 else f"{base_topic} (aspect {asked + 1})"
    docs = _get_retriever().invoke(query)
    if not docs:
        return {"context": "", "source_citations": []}
    ctx = "\n\n".join(d.page_content for d in docs)
    citations = []
    for d in docs:
        raw = d.metadata.get("source", "?")
        filename = Path(raw).name if raw != "?" else "?"
        page = d.metadata.get("page", "?")
        citations.append(f"{filename} (page {page})")
    # Deduplicate while preserving order
    seen = set()
    unique_citations = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            unique_citations.append(c)
    return {"context": ctx, "source_citations": unique_citations}


def generate_question(state: QuizState) -> dict:
    """
    Node: ask the LLM to write ONE question + the expected answer.
    Resets per-question fields (retries, hint, user_answer, verdict, feedback).

    Passes the last few questions from `state["history"]` to the prompt so the
    LLM can avoid repeating them. The prompt itself is responsible for
    picking a different sub-topic; this node just feeds it the context.

    Primary path: prompt | llm | JsonOutputParser validates against QuizQuestion.
    Fallback: if the parser rejects the LLM output, try _safe_json_loads as
    a last-ditch attempt before surfacing the error.
    """
    history = state.get("history", [])
    recent = history[-3:]  # last 3 questions
    if recent:
        prev_text = "Previous questions on this topic:\n" + "\n".join(
            f"- {entry['q']}" for entry in recent
        )
    else:
        prev_text = ""

    inputs = {
        "context": state["context"],
        "format_instructions": _question_format,
        "previous_questions": prev_text,
    }
    try:
        data = _question_chain.invoke(inputs)
    except OutputParserException:
        raw = (QUIZ_GEN_PROMPT | _llm).invoke(inputs).content
        data = _safe_json_loads(raw)

    return {
        "question": data["question"],
        "expected": data["expected_answer"],
        "user_answer": None,
        "verdict": None,
        "feedback": None,
        "hint": None,
        "retries": 0,
    }


def grade_answer(state: QuizState) -> dict:
    """
    Node: grade student's answer against expected + source.
    Verdict: correct | partial | incorrect.

    Primary path: prompt | llm | JsonOutputParser validates against GradeResult,
    which enforces verdict ∈ {correct, partial, incorrect}.
    Fallback: _safe_json_loads if the parser rejects the output.
    """
    inputs = {
        "question": state["question"],
        "expected": state["expected"],
        "answer": state["user_answer"],
        "context": state["context"],
        "format_instructions": _grade_format,
    }
    try:
        data = _grade_chain.invoke(inputs)
    except OutputParserException:
        raw = (GRADE_PROMPT | _llm).invoke(inputs).content
        data = _safe_json_loads(raw)

    verdict = data.get("verdict", "incorrect")
    if verdict not in {"correct", "partial", "incorrect"}:
        verdict = "incorrect"

    return {
        "verdict": verdict,
        "feedback": data.get("feedback", ""),
    }


def give_hint(state: QuizState) -> dict:
    """
    Node: nudge student toward the answer without revealing it.
    Bumps retries counter so we cap at one retry per question.

    Plain-text output (no JSON contract needed), so we use StrOutputParser.
    """
    text = _hint_chain.invoke({
        "question": state["question"],
        "answer": state["user_answer"],
        "context": state["context"],
    })
    return {
        "hint": text.strip(),
        "retries": state["retries"] + 1,
        "user_answer": None,
        "verdict": None,
        "feedback": None,
    }


def advance_question(state: QuizState) -> dict:
    """
    Node: log the result of the current question, update score and weak topics,
    bump asked_count, and clear per-question fields ready for the next round.

    `skipped` is True when the user picked "Skip to next" instead of
    "Get hint" after a wrong answer (signalled by retries >= 1 with a
    non-correct verdict and no actual retry attempt — i.e. no fresh
    user_answer was graded). Such entries count as wrong for scoring/weak
    topics but are marked in history for UI display.
    """
    correct = state["verdict"] == "correct"
    skipped = (
        not correct
        and state.get("retries", 0) >= 1
        and not state.get("user_answer")  # hint path clears user_answer; skip has none
    )
    return {
        "asked_count": state["asked_count"] + 1,
        "score": state["score"] + (1 if correct else 0),
        "history": state["history"] + [{
            "q": state["question"],
            "expected": state["expected"],
            "user_answer": state.get("user_answer"),
            "verdict": state["verdict"],
            "topic": state["topic"],
            "sources": list(state.get("source_citations", [])),
            "skipped": skipped,
        }],
        "weak_topics": (
            state["weak_topics"] + [state["topic"]]
            if not correct
            else state["weak_topics"]
        ),
        # Clear per-question fields
        "question": None,
        "expected": None,
        "user_answer": None,
        "verdict": None,
        "feedback": None,
        "hint": None,
        "retries": 0,
        "source_citations": [],
        "awaiting_continue": False,
    }


# ---------- Routers ----------

def route_after_grade(state: QuizState) -> Literal["hint", "advance"]:
    """
    After grading, decide: give a hint (and let the student retry) or move on.
    - correct          -> advance (no retry needed)
    - already retried  -> advance (cap at 1 retry per question)
    - otherwise        -> hint, let them retry
    """
    if state["verdict"] == "correct":
        return "advance"
    if state["retries"] >= 1:
        return "advance"
    return "hint"


def route_after_advance(state: QuizState) -> Literal["new_question", "end"]:
    """After advancing: another question, or finish the quiz."""
    if state["asked_count"] >= state["max_questions"]:
        return "end"
    return "new_question"


# ---------- Compiled graph (Day 3 upgrade) ----------
#
# Wire the same node functions into a real StateGraph. Two pause points
# are needed for the quiz to work as an interactive loop:
#
#   1. before grade_answer (initial answer)
#   2. before grade_answer again (retry after hint)
#
# A single interrupt_before=["grade_answer"] handles both, because the
# grade_answer node is reached from two different incoming edges:
#   generate_question -> grade_answer   (first attempt)
#   give_hint         -> grade_answer   (retry)
#
# Streamlit drives the graph like this:
#   - first run: graph.invoke(initial_state, config)  -> pauses before grade
#   - after user submits answer:
#       graph.update_state(config, {"user_answer": "..."})
#       graph.invoke(None, config)                     -> resumes, grades,
#                                                        either pauses again
#                                                        (hint path) or
#                                                        advances + asks
#                                                        the next question


def _build_quiz_graph():
    """
    Build and compile the quiz StateGraph with a SQLite checkpointer.

    The checkpointer keeps state per `thread_id` so the same conversation
    can be paused and resumed across multiple .invoke() calls. Streamlit
    treats each quiz session as one thread_id.

    SQLite (via SqliteSaver) means in-progress quizzes survive an app
    restart — a half-finished quiz can be resumed by calling graph.invoke
    with the same thread_id.
    """
    g = StateGraph(QuizState)

    # Nodes — same functions as the imperative path.
    g.add_node("retrieve", retrieve_topic_content)
    g.add_node("generate", generate_question)
    g.add_node("grade", grade_answer)
    g.add_node("hint", give_hint)
    g.add_node("advance", advance_question)

    # Static edges
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "grade")          # interrupt fires before grade
    g.add_edge("hint", "grade")              # retry path — interrupt fires again

    # Conditional edges
    g.add_conditional_edges(
        "grade",
        route_after_grade,
        {"hint": "hint", "advance": "advance"},
    )
    g.add_conditional_edges(
        "advance",
        route_after_advance,
        {"new_question": "retrieve", "end": END},
    )

    # Use SQLite checkpointer (persists across app restarts) under
    # user_chroma_db/. Falls back to MemorySaver only if the SQLite path
    # can't be created (e.g. permissions issue).
    db_path = ACTIVE_CHROMA_DIR / "langgraph_checkpoints.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(str(db_path), check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()  # create tables if missing
    except Exception:
        checkpointer = MemorySaver()

    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=["grade"],
        interrupt_after=["grade"],
    )


# Build once at import time. The graph object is reusable across all
# threads — only the checkpointed state is per-thread.
QUIZ_GRAPH = _build_quiz_graph()


def get_quiz_graph():
    """Public accessor in case future code wants to rebuild with a different checkpointer."""
    return QUIZ_GRAPH
