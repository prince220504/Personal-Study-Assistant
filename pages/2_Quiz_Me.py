"""
Quiz Me page — Streamlit UI on top of the quiz_graph node functions.

Day 2 design (option A from IMPLEMENTATION_PLAN.md section 4):
We do NOT use a compiled LangGraph here. Instead we call the node
functions imperatively from this page and stash the QuizState in
st.session_state. Streamlit's own rerun model handles the "pause for
user input" part naturally.

Day 3 upgrade: replace this with a compiled StateGraph that uses
MemorySaver + interrupt_before=["grade_answer"].
"""

import streamlit as st

from src.quiz_graph import (
    make_initial_state,
    retrieve_topic_content,
    generate_question,
    grade_answer,
    give_hint,
    advance_question,
    route_after_grade,
    route_after_advance,
)


# ---------- Page setup ----------

st.title("🧠 Quiz Me")
st.caption("Pick a topic from your notes. The assistant generates questions, grades your answers, and tracks weak spots.")


# ---------- Initialise session state ----------

if "quiz" not in st.session_state:
    st.session_state.quiz = None


# ---------- Helper: start the next question ----------

def _start_next_question():
    """Retrieve fresh context for the topic and generate the next question."""
    s = st.session_state.quiz
    s.update(retrieve_topic_content(s))
    if not s["context"]:
        st.error(
            f"No chunks found for topic '{s['topic']}'. "
            "Try a topic that matches words used in your notes."
        )
        st.session_state.quiz = None
        st.stop()
    s.update(generate_question(s))
    st.session_state.quiz = s


# ---------- Screen 1: setup ----------

if st.session_state.quiz is None:
    st.subheader("Start a new quiz")

    topic = st.text_input(
        "Topic",
        placeholder="e.g. embeddings, transformers, RAG",
    )
    n = st.number_input(
        "How many questions?",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    if st.button("Start quiz", type="primary", disabled=not topic.strip()):
        st.session_state.quiz = make_initial_state(topic.strip(), int(n))
        with st.spinner("Picking a question for you..."):
            _start_next_question()
        st.rerun()


# ---------- Screen 2: active quiz ----------

else:
    s = st.session_state.quiz

    # ---- Final-score screen ----
    if s["asked_count"] >= s["max_questions"]:
        st.subheader("Quiz complete")
        st.metric("Score", f"{s['score']} / {s['max_questions']}")

        weak = sorted(set(s["weak_topics"]))
        if weak:
            st.markdown("### Weak topics")
            for t in weak:
                st.markdown(f"- {t}")
        else:
            st.success("No weak topics — great job!")

        with st.expander("Review your answers"):
            for i, entry in enumerate(s["history"], 1):
                st.markdown(f"**Q{i} ({entry['verdict']}):** {entry['q']}")
                st.markdown(f"- Your answer: {entry['user_answer']}")
                st.markdown(f"- Expected: {entry['expected']}")
                st.divider()

        if st.button("Start over"):
            st.session_state.quiz = None
            st.rerun()

    # ---- Active question screen ----
    else:
        # Progress
        st.progress(s["asked_count"] / s["max_questions"])
        st.caption(f"Question {s['asked_count'] + 1} of {s['max_questions']}  ·  Topic: {s['topic']}")

        st.markdown(f"### {s['question']}")

        # Persistent feedback banner — survives reruns
        if s.get("verdict"):
            verdict = s["verdict"]
            feedback = s.get("feedback", "")
            if verdict == "correct":
                st.success(f"✅ Correct — {feedback}")
            elif verdict == "partial":
                st.warning(f"🟡 Partial — {feedback}")
            else:
                st.error(f"❌ Incorrect — {feedback}")

            # Show what they answered so they can compare
            if s.get("user_answer"):
                with st.expander("Your answer"):
                    st.write(s["user_answer"])

        # Show hint (if previous attempt was wrong and they can retry)
        if s.get("hint"):
            st.info(f"💡 Hint: {s['hint']}")

        # ---- Branch: waiting for "Continue" vs accepting new answer ----
        if s.get("awaiting_continue"):
            # User has read the verdict. Move on when ready.
            if st.button("Continue →", type="primary"):
                s.update(advance_question(s))
                st.session_state.quiz = s
                if s["asked_count"] < s["max_questions"]:
                    with st.spinner("Picking the next question..."):
                        _start_next_question()
                st.rerun()

            if st.button("Quit quiz"):
                st.session_state.quiz = None
                st.rerun()

        else:
            # Accept a (possibly retry) answer
            answer_key = f"answer_{s['asked_count']}_{s['retries']}"
            user_answer = st.text_area(
                "Your answer",
                key=answer_key,
                height=100,
            )

            col1, col2 = st.columns([1, 5])
            with col1:
                submit = st.button("Submit", type="primary", disabled=not user_answer.strip())
            with col2:
                quit_btn = st.button("Quit quiz")

            if quit_btn:
                st.session_state.quiz = None
                st.rerun()

            if submit:
                s["user_answer"] = user_answer.strip()
                with st.spinner("Grading..."):
                    s.update(grade_answer(s))

                next_step = route_after_grade(s)
                if next_step == "hint":
                    # Wrong, can still retry — show hint and let them try again
                    with st.spinner("Thinking of a hint..."):
                        s.update(give_hint(s))
                    s["awaiting_continue"] = False
                else:
                    # Correct, or out of retries — pause for user to read before advancing
                    s["awaiting_continue"] = True

                st.session_state.quiz = s
                st.rerun()
