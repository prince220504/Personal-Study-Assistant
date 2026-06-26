from uuid import uuid4

import streamlit as st

from src.quiz_graph import get_quiz_graph, make_initial_state
from src.upload_ui import render_pdf_upload_panel, list_uploaded_pdfs


st.title("🧠 Quiz Me")
st.caption("Quiz yourself from your uploaded PDFs. LangGraph controls the quiz state and retry flow.")

ready = render_pdf_upload_panel()

if not ready:
    st.warning("Upload one or more PDFs and build the index before starting a quiz.")
    st.stop()

if "quiz_thread_id" not in st.session_state:
    st.session_state.quiz_thread_id = None
if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = None


def _graph_config():
    return {"configurable": {"thread_id": st.session_state.quiz_thread_id}}


def _reset_quiz():
    st.session_state.quiz_thread_id = None
    st.session_state.quiz_state = None


def _start_quiz(topic, max_questions):
    graph = get_quiz_graph()
    st.session_state.quiz_thread_id = f"quiz-{uuid4()}"
    initial_state = make_initial_state(topic, max_questions)
    st.session_state.quiz_state = graph.invoke(initial_state, _graph_config())


def _submit_answer(answer):
    graph = get_quiz_graph()
    graph.update_state(_graph_config(), {"user_answer": answer})
    st.session_state.quiz_state = graph.invoke(None, _graph_config())


def _continue_after_feedback():
    graph = get_quiz_graph()
    st.session_state.quiz_state = graph.invoke(None, _graph_config())


# ---------- Quiz setup screen ----------

if st.session_state.quiz_state is None:
    st.subheader("Start a new quiz")

    # Topic dropdown from uploaded PDF filenames + free-text option
    pdfs = list_uploaded_pdfs()
    pdf_names = [p.stem for p in pdfs]  # filename without extension

    topic_options = ["(type your own)"] + pdf_names
    selected_option = st.selectbox(
        "Pick a topic from your PDFs or type your own:",
        options=topic_options,
    )

    if selected_option == "(type your own)":
        topic = st.text_input(
            "Topic",
            placeholder="e.g. summary, chapter 1, transformers, evaluation metrics",
        )
    else:
        topic = selected_option

    max_questions = st.number_input(
        "How many questions?",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    if st.button("Start quiz", type="primary", disabled=not topic.strip()):
        with st.spinner("LangGraph is retrieving context and generating a question..."):
            _start_quiz(topic.strip(), int(max_questions))
        st.rerun()

    st.stop()


# ---------- Active quiz ----------

state = st.session_state.quiz_state

if not state.get("context"):
    st.error(
        f"No useful chunks were retrieved for topic '{state['topic']}'. "
        "Try a topic or keyword that appears in your uploaded PDFs."
    )
    if st.button("Start over"):
        _reset_quiz()
        st.rerun()
    st.stop()

# ---------- Quiz complete screen ----------

if state["asked_count"] >= state["max_questions"]:
    st.subheader("Quiz complete")
    st.metric("Score", f"{state['score']} / {state['max_questions']}")

    weak = sorted(set(state["weak_topics"]))
    if weak:
        st.markdown("### Weak topics")
        for topic in weak:
            st.markdown(f"- {topic}")
    else:
        st.success("No weak topics. Nice work.")

    with st.expander("Review your answers"):
        for i, entry in enumerate(state["history"], 1):
            st.markdown(f"**Q{i} ({entry['verdict']}):** {entry['q']}")
            st.markdown(f"- Your answer: {entry['user_answer']}")
            st.markdown(f"- Expected: {entry['expected']}")
            sources = entry.get("sources", [])
            if sources:
                st.caption(f"📄 Sources: {', '.join(sources)}")
            st.divider()

    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("Start over"):
            _reset_quiz()
            st.rerun()
    with col2:
        # Re-quiz weak topics button
        if weak:
            if st.button("Re-quiz weak topics", type="primary"):
                weak_topic_str = ", ".join(weak)
                with st.spinner(f"Starting quiz on weak topics: {weak_topic_str}..."):
                    _reset_quiz()
                    _start_quiz(weak_topic_str, min(len(weak) * 2, 10))
                st.rerun()

    st.stop()


# ---------- Question in progress ----------

st.progress(state["asked_count"] / state["max_questions"])
st.caption(
    f"Question {state['asked_count'] + 1} of {state['max_questions']} | "
    f"Topic: {state['topic']} | Graph thread: {st.session_state.quiz_thread_id}"
)

st.markdown(f"### {state['question']}")

# Source citations for the current question
citations = state.get("source_citations", [])
if citations:
    st.caption(f"📄 Sources: {', '.join(citations)}")

if state.get("verdict"):
    verdict = state["verdict"]
    feedback = state.get("feedback", "")
    if verdict == "correct":
        st.success(f"Correct — {feedback}")
    elif verdict == "partial":
        st.warning(f"Partial — {feedback}")
    else:
        st.error(f"Incorrect — {feedback}")

    if state.get("user_answer"):
        with st.expander("Your previous answer"):
            st.write(state["user_answer"])

if state.get("hint"):
    st.info(f"Hint: {state['hint']}")

if state.get("verdict"):
    col1, col2 = st.columns([1, 5])
    with col1:
        continue_quiz = st.button("Continue", type="primary")
    with col2:
        quit_quiz = st.button("Quit quiz")

    if quit_quiz:
        _reset_quiz()
        st.rerun()

    if continue_quiz:
        with st.spinner("LangGraph is routing the next step..."):
            _continue_after_feedback()
        st.rerun()

else:
    answer_key = f"answer_{st.session_state.quiz_thread_id}_{state['asked_count']}_{state['retries']}"
    user_answer = st.text_area("Your answer", key=answer_key, height=100)

    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("Submit", type="primary", disabled=not user_answer.strip())
    with col2:
        quit_quiz = st.button("Quit quiz")

    if quit_quiz:
        _reset_quiz()
        st.rerun()

    if submit:
        with st.spinner("LangGraph is grading your answer..."):
            _submit_answer(user_answer.strip())
        st.rerun()
