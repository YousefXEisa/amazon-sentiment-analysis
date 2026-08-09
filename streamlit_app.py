import streamlit as st
from src.inference import SentimentPredictor


st.set_page_config(
    page_title="Amazon Reviews Sentiment Analysis",
    page_icon="🛍️",
    layout="centered"
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .stApp, [data-testid="stHeader"] { background-color: #0b0f19; }

        [data-testid="stForm"] {
            background-color: #11151f;
            border: 1px solid #2a2f3a;
            border-radius: 14px;
            padding: 1.75rem;
        }

        h1 {
            color: #f4f4f5;
            font-size: clamp(1.3rem, 3.2vw, 1.9rem);
            white-space: nowrap;
        }

        p, .stCaption, label { color: #9ca3af !important; }

        .stTextInput input, .stTextArea textarea {
            background-color: #0b0f19;
            color: #f4f4f5;
            border: 1px solid #2a2f3a;
            border-radius: 8px;
        }

        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #7c3aed;
            box-shadow: 0 0 0 1px #7c3aed;
        }

        div.stButton > button {
            background-color: #f4f4f5;
            color: #0b0f19;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.65rem 1rem;
            transition: background-color 0.2s ease;
        }

        div.stButton > button:hover { background-color: #d4d4d8; }

        hr { border-color: #2a2f3a; margin: 2rem 0 1rem 0; }

        .result-card {
            background-color: #11151f;
            border: 1px solid #2a2f3a;
            border-left: 4px solid var(--accent, #8b5cf6);
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            margin-top: 1.5rem;
            font-weight: 600;
            color: #f4f4f5;
        }

        .prob-wrapper { margin-top: 1.4rem; }
        .prob-row { margin-bottom: 0.9rem; }
        .prob-row:last-child { margin-bottom: 0; }

        .prob-row-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            margin-bottom: 4px;
        }
        .prob-row-header span:first-child { color: #d4d4d8; font-weight: 500; }
        .prob-row-header span:last-child { color: #9ca3af; }

        .prob-track {
            width: 100%;
            height: 6px;
            background-color: #2a2f3a;
            border-radius: 3px;
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            border-radius: 3px;
            background: linear-gradient(90deg, #7c3aed, #a78bfa);
            transition: width 0.7s ease;
        }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource(show_spinner="Loading model from Hugging Face Hub...")
def load_predictor():
    return SentimentPredictor()


def render_bar(label, pct):
    return (
        f'<div class="prob-row">'
        f'<div class="prob-row-header"><span>{label}</span><span>{pct:.2f}%</span></div>'
        f'<div class="prob-track"><div class="prob-fill" style="width:{pct:.2f}%;"></div></div>'
        f'</div>'
    )


st.title("Amazon Reviews Sentiment Analysis")
st.caption("RoBERTa model fine-tuned on Amazon reviews — classifies text as Positive or Negative.")

predictor = load_predictor()

with st.form("predict_form"):
    title = st.text_input("Review Title (Optional)", placeholder="Great product!")
    content = st.text_area(
        "Review Content",
        placeholder="I've been using this for a month and it works perfectly...",
        height=120
    )
    submitted = st.form_submit_button("Predict", use_container_width=True)


if submitted:
    if not title.strip() and not content.strip():
        st.error("Please provide at least a review title or review content.")
    else:
        with st.spinner("Analyzing review..."):
            try:
                result = predictor.predict(title=title, content=content)
                label = result["label"]
                probs = result["probabilities"]
                order = ["Positive score", "Negative score"]
                probs = {k: probs[k] for k in order if k in probs}

                accent = "#8b5cf6"
                bars = "".join(render_bar(k, v * 100) for k, v in probs.items())

                st.markdown(
                    f'<div class="result-card" style="--accent:{accent};">Result: {label}</div>'
                    f'<div class="prob-wrapper">{bars}</div>',
                    unsafe_allow_html=True
                )

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")


st.markdown("---")
st.caption(
    "This demo runs on the fine-tuned model directly. "
    "A local Gradio interface and a production-ready FastAPI service are also included — see the README for setup."
)