from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template("""\
Answer the question using only the context below. If the context does not contain the answer, say so.

Context:
{context}

Question: {question}

Answer:""")

QUIZ_GEN_PROMPT = ChatPromptTemplate.from_template("""\
You are a quiz generator. From the study material below, write ONE clear question testing understanding (not trivia).

Return your response as a JSON object with exactly these keys: "question" and "expected_answer".
Do NOT include markdown formatting.

Material:
{context}""")

GRADE_PROMPT = ChatPromptTemplate.from_template("""\
Grade the student's answer against the expected answer and source material.

Return your response as a JSON object with exactly these keys: "verdict" (one of: correct, partial, incorrect) and "feedback" (short explanation).
Do NOT include markdown formatting.

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
