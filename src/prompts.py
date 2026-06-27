from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# RAG prompt — adaptive structure based on question type.
# The LLM picks the format from the question text.
# ---------------------------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_template("""\
You are a friendly study assistant helping a student understand their notes.

Use ONLY the context below as your source of truth. Do not invent facts.
If the context does not contain the answer, say so clearly and briefly.

Pick the structure that best fits the question. Read the question carefully first.

OPTION A — Short definitional question (e.g. "What is X?", "Define X", "What does X mean?"):
Use this exact 5-section structure with markdown headings:

### Definition (from notes)
> Quote the exact sentence(s) from the context that define or describe the concept. Use a markdown blockquote (>). Keep the wording verbatim.

### In Simple Words
Rewrite that definition in plain, beginner-friendly language (2-3 sentences). Break down any jargon.

### Key Points
- 2-4 short bullets covering the main ideas from the context

### Example
One concrete example or analogy from the context, or a simple accurate one based on it.

### Why It Matters
1-2 sentences on why this concept is useful or where it shows up.

OPTION B — Explanatory / comparison / "in detail" question (e.g. "Explain X in detail", "How does X work?", "Compare X and Y", "Walk me through X"):
Use flowing prose with H2/H3 subheadings. Do NOT use the strict 5-section format. Cover the topic thoroughly: what it is, how it works, key components, when to use it, common pitfalls. Adapt the subheadings to the question. Use code blocks for code, bullet lists for parallel ideas, prose for narrative. Length should match the question — "in detail" gets more text than "briefly explain".

OPTION C — How-to / step-by-step question (e.g. "How do I...?", "Steps to...", "Show me how to..."):
Use a numbered list of steps. After each step, add one short sentence explaining why. End with a short "Common mistakes" section if relevant.

If the question is ambiguous, prefer Option B. If the user explicitly asks for brevity, keep your answer short.

Context:
{context}

Question: {question}

Answer:""")


# ---------------------------------------------------------------------------
# Quiz generation prompt — history-aware to avoid repeating questions.
# ---------------------------------------------------------------------------

QUIZ_GEN_PROMPT = ChatPromptTemplate.from_template("""\
You are a quiz generator. From the study material below, write ONE clear question testing understanding (not trivia).

{format_instructions}

If previous questions on this topic are listed below, ask about a DIFFERENT aspect — do not repeat or paraphrase any of them. Aim to cover a new sub-concept, edge case, or application of the material.

{previous_questions}

Material:
{context}""")


# ---------------------------------------------------------------------------
# Grading prompt — forgiving. Accept paraphrased correct answers.
# ---------------------------------------------------------------------------

GRADE_PROMPT = ChatPromptTemplate.from_template("""\
You are grading a student's answer to a study question. Be precise about SUBSTANCE but forgiving about WORDING.

Step 1 — Identify the MAIN points of the expected answer. A typical expected answer has 1-3 MAIN points. A main point is a distinct idea (a fact, a mechanism, a benefit, a use-case). Supporting details, examples, and rephrasing of the same idea are NOT separate main points.

Step 2 — Check which main points the student's answer covers (in their own words).

Step 3 — Assign the verdict:

Mark "correct" if the student covers ALL main points, even when:
- they use different words (paraphrasing is fine)
- they use different examples
- they add extra information
- the order or structure differs

Mark "partial" if the student covers SOME but NOT ALL main points:
- misses one significant aspect the expected answer covers
- the answer is materially incomplete
- they demonstrate understanding of part of the concept but not the whole

Mark "incorrect" if the student covers NONE of the main points:
- contradicts the expected answer
- is off-topic
- is too vague to demonstrate understanding (e.g. "it's a thing used in code")
- explicitly says "I don't know", leaves it blank, or just restates the question

Examples:
- Expected: "A decorator wraps a function to extend its behavior. It is used for logging, caching, and access control." Student: "A decorator wraps a function and adds extra behavior." → correct (covers both main points; "logging/caching" is a supporting example)
- Expected: same. Student: "A decorator wraps a function." → partial (missing the "extends behavior / use cases" point)
- Expected: same. Student: "A decorator adds caching to functions." → incorrect (wrong mechanism entirely)
- Expected: same. Student: "It is used in Python code." → incorrect (too vague)

{format_instructions}

Question: {question}
Expected answer: {expected}
Student answer: {answer}
Source context: {context}""")


# ---------------------------------------------------------------------------
# Hint prompt — plain text, no JSON.
# ---------------------------------------------------------------------------

HINT_PROMPT = ChatPromptTemplate.from_template("""\
The student got this question wrong or partially correct. Give ONE short hint that nudges them toward the answer without giving it away completely.

Question: {question}
Their answer: {answer}
Source context: {context}

Hint:""")
