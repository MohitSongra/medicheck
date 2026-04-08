"""
MediCheck - Symptom Input Page
===============================
Allows users to select symptoms from categorised groups and
submit them to the FastAPI backend for analysis.
Dynamically builds categories from the 132-symptom Kaggle dataset.
"""

import streamlit as st
import httpx

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Symptom categories for better UX (132 Kaggle symptoms organised)
# ---------------------------------------------------------------------------
SYMPTOM_CATEGORIES = {
    "🌡️ General / Systemic": [
        "fatigue", "high_fever", "mild_fever", "chills", "shivering",
        "sweating", "dehydration", "malaise", "lethargy", "restlessness",
        "weight_loss", "weight_gain", "loss_of_appetite", "increased_appetite",
        "obesity", "toxic_look_(typhos)", "altered_sensorium", "coma",
    ],
    "🫁 Respiratory": [
        "cough", "breathlessness", "phlegm", "mucoid_sputum", "rusty_sputum",
        "blood_in_sputum", "throat_irritation", "patches_in_throat",
        "continuous_sneezing", "runny_nose", "congestion", "sinus_pressure",
        "chest_pain",
    ],
    "🧠 Neurological / Mental": [
        "headache", "dizziness", "spinning_movements", "loss_of_balance",
        "unsteadiness", "weakness_of_one_body_side", "lack_of_concentration",
        "visual_disturbances", "blurred_and_distorted_vision", "slurred_speech",
        "loss_of_smell", "anxiety", "depression", "irritability", "mood_swings",
    ],
    "🦴 Musculoskeletal": [
        "joint_pain", "knee_pain", "hip_joint_pain", "neck_pain",
        "back_pain", "muscle_weakness", "muscle_wasting", "muscle_pain",
        "stiff_neck", "swelling_joints", "movement_stiffness",
        "weakness_in_limbs", "cramps", "painful_walking",
    ],
    "🫄 Gastrointestinal": [
        "stomach_pain", "acidity", "ulcers_on_tongue", "vomiting",
        "nausea", "diarrhoea", "constipation", "abdominal_pain",
        "belly_pain", "indigestion", "passage_of_gases",
        "stomach_bleeding", "distention_of_abdomen",
    ],
    "💓 Cardiovascular": [
        "fast_heart_rate", "palpitations", "swollen_blood_vessels",
        "swollen_legs", "prominent_veins_on_calf", "cold_hands_and_feets",
    ],
    "🧪 Urinary / Metabolic": [
        "burning_micturition", "spotting_urination", "dark_urine",
        "yellow_urine", "bladder_discomfort", "foul_smell_of_urine",
        "continuous_feel_of_urine", "polyuria", "irregular_sugar_level",
        "excessive_hunger",
    ],
    "🩹 Skin / Dermatological": [
        "skin_rash", "nodal_skin_eruptions", "itching", "internal_itching",
        "yellowish_skin", "dischromic_patches", "pus_filled_pimples",
        "blackheads", "scurring", "skin_peeling", "silver_like_dusting",
        "small_dents_in_nails", "inflammatory_nails", "blister",
        "red_sore_around_nose", "yellow_crust_ooze", "bruising",
        "red_spots_over_body",
    ],
    "👁️ Eyes / ENT": [
        "sunken_eyes", "yellowing_of_eyes", "redness_of_eyes",
        "watering_from_eyes", "pain_behind_the_eyes",
    ],
    "🫀 Liver / Hepatic": [
        "yellowish_skin", "dark_urine", "acute_liver_failure",
        "fluid_overload", "swelling_of_stomach",
        "history_of_alcohol_consumption",
    ],
    "🔬 Blood / Immune": [
        "swelled_lymph_nodes", "receiving_blood_transfusion",
        "receiving_unsterile_injections", "extra_marital_contacts",
        "family_history",
    ],
    "🦵 Other": [
        "pain_during_bowel_movements", "pain_in_anal_region",
        "bloody_stool", "irritation_in_anus",
        "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails",
        "swollen_extremeties", "drying_and_tingling_lips",
        "abnormal_menstruation",
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
        "Choose the symptoms you are currently experiencing from the categories below. "
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

    # Deduplicate (some symptoms appear in multiple categories)
    selected = list(dict.fromkeys(selected))

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
