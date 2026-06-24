from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .ingest import build_vectorstore
from .prompts import RAG_PROMPT
from .config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, TOP_K


def format_docs(docs):
    """Convert retrieved chunks into a single string for the prompt."""
    parts = []
    for i, d in enumerate(docs, 1):
        raw_source = d.metadata.get("source", "?")
        filename = Path(raw_source).name if raw_source != "?" else "?"
        page = d.metadata.get("page", "?")
        parts.append(f"[Source {i}: {filename} (page {page})]\n{d.page_content}")
    return "\n\n".join(parts)


def build_rag_chain():
    """
    Build LCEL chain: retrieve → format → prompt → LLM → string output.

    Returns (chain, retriever). Retriever needed separately to show
    source chunks in the UI.
    """
    vs = build_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=LLM_TEMPERATURE,
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever
