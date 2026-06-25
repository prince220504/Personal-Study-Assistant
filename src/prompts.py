from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template("""\
You are a friendly study assistant helping a student understand their notes.

Use ONLY the context below as your source of truth. Do not invent facts.
If the context does not contain the answer, say so clearly.

Structure your answer in markdown like this:

### Definition (from notes)
> Quote the exact sentence(s) from the context that define or describe the concept. Use a markdown blockquote (>). Keep the wording verbatim.

### In Simple Words
Rewrite that definition in plain, beginner-friendly language (2-3 sentences). Break down any jargon. This is the "what it really means" part.

### Key Points
- 2-4 short bullets covering the main ideas from the context

### Example
One concrete example or analogy. Use the example from the context if there is one; otherwise create a simple, accurate one based on the context.

### Why It Matters
1-2 sentences on why this concept is useful or where it shows up.

Context:
{context}

Question: {question}

Answer:""")

QUIZ_GEN_PROMPT = ChatPromptTemplate.from_template("""\
You are a quiz generator. From the study material below, write ONE clear question testing understanding (not trivia).

{format_instructions}

Material:
{context}""")

GRADE_PROMPT = ChatPromptTemplate.from_template("""\
Grade the student's answer against the expected answer and source material.

{format_instructions}

Question: {question}
Expected answer: {expected}
Student answer: {answer}
Source context: {context}""")

HINT_PROMPT = ChatPromptTemplate.from_template("""\
The student got this question wrong or partially correct. Give ONE short hint that nudges them toward the answer without giving it away completely.

Question: {question}
Their answer: {answer}
Source context: {context}

Hint:""")
