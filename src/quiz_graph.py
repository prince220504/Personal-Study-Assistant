"""
Quiz Graph — LangGraph-style state machine for the Quiz Me feature.

Day 2 implementation uses "option A": we define the State TypedDict and the
node functions, but we DO NOT compile a LangGraph yet. The Streamlit page
calls these node functions directly between reruns, and st.session_state
holds the QuizState dict.

Day 3 upgrade path: build a compiled StateGraph with a MemorySaver
checkpointer and `interrupt_before=["grade_answer"]` so the graph pauses
naturally for user input.

Concepts used:
- State: shared dict every node reads/writes
- Node: function (state) -> partial state update
- Conditional routing: function (state) -> next-node name
"""

import json
import re
from typing import TypedDict, List, Optional, Literal

from langchain_groq import ChatGroq

from .ingest import build_vectorstore
from .prompts import QUIZ_GEN_PROMPT, GRADE_PROMPT, HINT_PROMPT
from .config import GROQ_API_KEY, LLM_MODEL, TOP_K


# ---------- State ----------

class QuizState(TypedDict):
    """
    The 'backpack' the quiz agent carries through the flow.
    Every node can read from it and return a partial update.
    """
    topic: str                       # what the user wants to be quizzed on
    context: str                     # retrieved chunks joined as one string
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
# Built once at import time. Reused across all node calls.

_llm = ChatGroq(
    model=LLM_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.3,
)
_vs = build_vectorstore()
_retriever = _vs.as_retriever(search_kwargs={"k": TOP_K})


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
    """
    docs = _retriever.invoke(state["topic"])
    if not docs:
        return {"context": ""}
    ctx = "\n\n".join(d.page_content for d in docs)
    return {"context": ctx}


def generate_question(state: QuizState) -> dict:
    """
    Node: ask the LLM to write ONE question + the expected answer.
    Resets per-question fields (retries, hint, user_answer, verdict, feedback).
    """
    raw = (QUIZ_GEN_PROMPT | _llm).invoke({"context": state["context"]}).content
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
    """
    raw = (GRADE_PROMPT | _llm).invoke({
        "question": state["question"],
        "expected": state["expected"],
        "answer": state["user_answer"],
        "context": state["context"],
    }).content
    data = _safe_json_loads(raw)
    return {
        "verdict": data["verdict"],
        "feedback": data["feedback"],
    }


def give_hint(state: QuizState) -> dict:
    """
    Node: nudge student toward the answer without revealing it.
    Bumps retries counter so we cap at one retry per question.
    """
    raw = (HINT_PROMPT | _llm).invoke({
        "question": state["question"],
        "answer": state["user_answer"],
        "context": state["context"],
    }).content
    return {
        "hint": raw.strip(),
        "retries": state["retries"] + 1,
    }


def advance_question(state: QuizState) -> dict:
    """
    Node: log the result of the current question, update score and weak topics,
    bump asked_count, and clear per-question fields ready for the next round.
    """
    correct = state["verdict"] == "correct"
    return {
        "asked_count": state["asked_count"] + 1,
        "score": state["score"] + (1 if correct else 0),
        "history": state["history"] + [{
            "q": state["question"],
            "expected": state["expected"],
            "user_answer": state["user_answer"],
            "verdict": state["verdict"],
            "topic": state["topic"],
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
