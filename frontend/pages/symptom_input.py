"""
MediCheck - Symptom Input Page
===============================
Allows users to select symptoms from categorised groups and
submit them to the FastAPI backend for analysis.
"""

import streamlit as st
import httpx

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Symptom categories for better UX
# ---------------------------------------------------------------------------
SYMPTOM_CATEGORIES = {
    "🌡️ General / Systemic": [
        "fever", "fatigue", "weakness", "chills", "sweating",
        "weight_loss", "night_sweats", "loss_of_appetite",
    ],
    "🫁 Respiratory": [
        "cough", "shortness_of_breath", "wheezing", "chest_tightness",
        "blood_in_sputum", "sore_throat", "runny_nose", "sneezing",
        "nasal_congestion",
    ],
    "🧠 Neurological": [
        "headache", "dizziness", "blurred_vision", "light_sensitivity",
        "stiff_neck", "loss_of_smell", "loss_of_taste",
    ],
    "🦴 Musculoskeletal": [
        "body_ache", "muscle_pain", "joint_pain", "back_pain",
    ],
    "🫄 Gastrointestinal": [
        "nausea", "vomiting", "diarrhea", "abdominal_pain",
        "constipation",
    ],
    "💓 Cardiovascular": [
        "chest_pain", "rapid_heartbeat", "high_blood_pressure",
        "pale_skin",
    ],
    "🧪 Urinary / Metabolic": [
        "frequent_urination", "burning_urination", "dark_urine",
        "increased_thirst", "dry_mouth", "excessive_hunger",
    ],
    "🩹 Skin / Other": [
        "skin_rash", "itching", "yellow_skin", "bruising",
        "swelling", "swollen_lymph_nodes", "red_eyes",
        "watery_eyes", "facial_pain",
    ],
}


def _format(name: str) -> str:
    return name.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render_symptom_input():
    st.markdown("### 🔍 Select Your Symptoms")
    st.markdown(
        "<p style='color:#666; margin-bottom:1.5rem;'>"
        "Choose the symptoms you are currently experiencing. "
        "Select at least one to proceed.</p>",
        unsafe_allow_html=True,
    )

    selected: list[str] = []

    # Render each category as an expander with checkboxes
    for category, symptoms in SYMPTOM_CATEGORIES.items():
        with st.expander(category, expanded=False):
            cols = st.columns(3)
            for idx, symptom in enumerate(symptoms):
                with cols[idx % 3]:
                    if st.checkbox(
                        _format(symptom),
                        key=f"sym_{symptom}",
                    ):
                        selected.append(symptom)

    # Summary & submit
    st.markdown("---")

    if selected:
        st.markdown(
            f"**Selected symptoms ({len(selected)}):** "
            + ", ".join(f"`{_format(s)}`" for s in selected)
        )
    else:
        st.info("👆 Please select at least one symptom above to begin analysis.")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyse_clicked = st.button(
            "🔬  Analyse Symptoms",
            disabled=len(selected) == 0,
            use_container_width=True,
            type="primary",
        )

    if analyse_clicked and selected:
        with st.spinner("🧬 Running Bayesian inference ..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/predict",
                    json={"symptoms": selected},
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()
                st.session_state.prediction_results = data
                st.success(
                    f"✅ Analysis complete in {data['inference_time_ms']:.1f} ms — "
                    "switch to **📊 Results** in the sidebar."
                )
            except httpx.ConnectError:
                st.error(
                    "❌ Could not connect to the API. "
                    "Make sure the backend is running:\n\n"
                    "```bash\n"
                    "cd medicheck\n"
                    "uvicorn api.main:app --reload\n"
                    "```"
                )
            except httpx.HTTPStatusError as exc:
                st.error(f"❌ API error: {exc.response.text}")
            except Exception as exc:
                st.error(f"❌ Unexpected error: {exc}")
