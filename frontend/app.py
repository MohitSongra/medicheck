"""
MediCheck - Streamlit Main Entry Point
=======================================
Handles the disclaimer gate and page navigation.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

APP_ENV = os.getenv("APP_ENV", "production")
DEBUG = os.getenv("DEBUG", "False") == "True"

import streamlit as st

# ---------------------------------------------------------------------------
# Page config  (MUST be first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MediCheck – AI Symptom Checker",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Medical White + Blue (#1A4D8F) theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Gradient header bar */
    .main-header {
        background: linear-gradient(135deg, #1A4D8F 0%, #2E7DD1 50%, #4DA3E8 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(26, 77, 143, 0.18);
    }
    .main-header h1 {
        color: #FFFFFF;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.88);
        font-size: 1.05rem;
        margin: 0.3rem 0 0 0;
        font-weight: 300;
    }

    /* Disclaimer card */
    .disclaimer-card {
        background: linear-gradient(145deg, #FFF8E1, #FFFDE7);
        border: 2px solid #F9A825;
        border-radius: 16px;
        padding: 2.5rem;
        max-width: 640px;
        margin: 4rem auto;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.08);
        text-align: center;
    }
    .disclaimer-card h2 {
        color: #E65100;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .disclaimer-card p {
        color: #424242;
        font-size: 1.05rem;
        line-height: 1.7;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D2B52 0%, #1A4D8F 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stRadio label span {
        color: #D0E4FF !important;
    }

    /* Card component */
    .info-card {
        background: #FFFFFF;
        border: 1px solid #E3EBF6;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(26, 77, 143, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(26, 77, 143, 0.12);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: #9E9E9E;
        font-size: 0.85rem;
        border-top: 1px solid #E0E0E0;
        margin-top: 3rem;
    }

    /* Hide Streamlit default menu/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False
if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "symptom_input"


# ---------------------------------------------------------------------------
# Disclaimer gate
# ---------------------------------------------------------------------------
def show_disclaimer():
    """Show a mandatory disclaimer that must be accepted to proceed."""
    st.markdown(
        """
        <div class="disclaimer-card">
            <h2>⚠️ Medical Disclaimer</h2>
            <p>
                <strong>This tool is for educational purposes only.<br>
                Not a substitute for professional medical advice.</strong>
            </p>
            <p style="font-size: 0.92rem; color: #616161; margin-top: 1rem;">
                MediCheck uses a Bayesian Network model trained on synthetic data.
                It is an academic project and must <em>never</em> be used for
                actual medical diagnosis. Always consult a qualified healthcare
                professional for any health concerns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button(
            "✅  I Understand",
            key="accept_disclaimer",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.disclaimer_accepted = True
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not st.session_state.disclaimer_accepted:
        show_disclaimer()
        return

    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1>🩺 MediCheck</h1>
            <p>AI-Powered Symptom Checker &nbsp;·&nbsp; Bayesian Network Inference</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Go to",
            ["🔍 Symptom Input", "📊 Results"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            "<p style='font-size:0.8rem; opacity:0.7;'>"
            "MediCheck v1.0 · Educational Use Only</p>",
            unsafe_allow_html=True,
        )

    # Route pages
    if page == "🔍 Symptom Input":
        from frontend.pages.symptom_input import render_symptom_input
        render_symptom_input()
    else:
        from frontend.pages.results import render_results
        render_results()


if __name__ == "__main__":
    main()
