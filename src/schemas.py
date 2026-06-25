"""
Pydantic schemas for structured LLM outputs.

These models serve two purposes:

1. They are passed to `JsonOutputParser` so LangChain can generate
   JSON-schema-style format instructions to inject into the prompt.
   The LLM sees the exact field names, types, and descriptions,
   which dramatically lowers the chance of a shape mismatch.

2. The parser validates the LLM's output against the schema before
   returning a dict. A missing field or wrong type raises an error
   we can catch and fall back from, instead of crashing deep in a
   node function with a KeyError.

Pydantic v2 is required (LangChain 0.3+ default). Do NOT import from
`langchain_core.pydantic_v1` here — that path is for the legacy v1 API.
"""

from typing import Literal
from pydantic import BaseModel, Field


class QuizQuestion(BaseModel):
    """Shape returned by the question-generation prompt."""

    question: str = Field(
        description="One clear question that tests understanding of the material, not trivia."
    )
    expected_answer: str = Field(
        description="The model answer in 1-3 sentences, grounded in the source material."
    )


class GradeResult(BaseModel):
    """Shape returned by the grading prompt."""

    verdict: Literal["correct", "partial", "incorrect"] = Field(
        description=(
            "Overall judgement of the student's answer. "
            "Use 'correct' for fully right, 'partial' for partially right, "
            "'incorrect' for wrong or off-topic."
        )
    )
    feedback: str = Field(
        description="One or two short sentences telling the student why they got that verdict."
    )
