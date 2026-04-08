"""
MediCheck - Results Page
=========================
Visualises disease predictions with colour-coded progress bars,
a Plotly pie chart, and detailed cards for each prediction.
"""

import streamlit as st
import plotly.graph_objects as go


def _confidence_color(pct: float) -> str:
    """Return Green / Yellow / Red based on confidence threshold."""
    if pct > 60:
        return "#2E7D32"   # Green
    elif pct >= 40:
        return "#F9A825"   # Yellow / Amber
    else:
        return "#C62828"   # Red


def _build_pie_chart(predictions: list[dict]) -> go.Figure:
    """Build a Plotly pie chart of confidence scores."""
    labels = [p["disease_name"] for p in predictions]
    values = [p["confidence_pct"] for p in predictions]
    colors = [_confidence_color(v) for v in values]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+percent",
                textfont=dict(size=13, family="Inter"),
                hovertemplate="<b>%{label}</b><br>Confidence: %{value:.1f}%<extra></extra>",
                hole=0.45,
            )
        ]
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        annotations=[
            dict(
                text="<b>Confidence</b>",
                x=0.5,
                y=0.5,
                font_size=15,
                showarrow=False,
                font_color="#1A4D8F",
            )
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render_results():
    st.markdown("### 📊 Analysis Results")

    data = st.session_state.get("prediction_results")

    if data is None:
        st.info(
            "No results yet. Go to **🔍 Symptom Input** to analyse your symptoms."
        )
        return

    predictions = data["predictions"]
    symptoms = data["symptoms_analyzed"]
    inference_ms = data["inference_time_ms"]

    # Metrics row
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Symptoms Analysed", len(symptoms))
    col_b.metric("Top Prediction", predictions[0]["disease_name"] if predictions else "—")
    col_c.metric("Inference Time", f"{inference_ms:.1f} ms")

    st.markdown("---")

    # Layout: progress bars left, pie chart right
    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### Confidence Rankings")
        for pred in predictions:
            pct = pred["confidence_pct"]
            color = _confidence_color(pct)
            label_color = "#FFFFFF" if pct > 15 else "#333333"

            st.markdown(
                f"""
                <div style="margin-bottom: 0.9rem;">
                    <div style="display:flex; justify-content:space-between;
                                align-items:center; margin-bottom:4px;">
                        <span style="font-weight:600; color:#1B1B1F;">
                            {pred['disease_name']}
                        </span>
                        <span style="font-weight:700; color:{color};">
                            {pct:.1f}%
                        </span>
                    </div>
                    <div style="background:#E8EEF6; border-radius:8px;
                                height:26px; overflow:hidden;">
                        <div style="width:{max(pct, 2):.1f}%; height:100%;
                                    background:{color}; border-radius:8px;
                                    display:flex; align-items:center;
                                    padding-left:10px; transition: width 0.6s ease;">
                            <span style="color:{label_color}; font-size:0.78rem;
                                         font-weight:500;">
                                {pct:.1f}%
                            </span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("#### Distribution")
        fig = _build_pie_chart(predictions)
        st.plotly_chart(fig, use_container_width=True)

    # Detailed cards
    st.markdown("---")
    st.markdown("#### 🩺 Detailed Breakdown")

    for pred in predictions:
        pct = pred["confidence_pct"]
        color = _confidence_color(pct)
        border_color = color

        st.markdown(
            f"""
            <div class="info-card" style="border-left: 5px solid {border_color};">
                <div style="display:flex; justify-content:space-between;
                            align-items:center; margin-bottom:0.5rem;">
                    <h4 style="margin:0; color:#1A4D8F;">
                        {pred['disease_name']}
                    </h4>
                    <span style="font-size:1.3rem; font-weight:700; color:{color};">
                        {pct:.1f}%
                    </span>
                </div>
                <p style="color:#555; margin:0.3rem 0;"><em>{pred['description']}</em></p>
                <p style="color:#1A4D8F; font-weight:500; margin:0.3rem 0;">
                    💡 {pred['suggested_action']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Disclaimer reminder
    st.markdown(
        """
        <div class="footer">
            ⚠️ <strong>Reminder:</strong> This tool is for educational purposes only.
            Always consult a qualified healthcare professional.
        </div>
        """,
        unsafe_allow_html=True,
    )
