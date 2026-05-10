import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from simulation_engine import run_simulation

st.set_page_config(page_title="Patient Dashboard", layout="wide")

st.title("AI Robotic Hand Rehabilitation")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Patient Profile")
patient_name = st.sidebar.text_input("Patient Name", "Nidhi")
age = st.sidebar.number_input("Age", 18, 90, 25)
severity = st.sidebar.selectbox("Severity Level", ["severe", "moderate", "mild"])

st.sidebar.header("Session Setup")
target_angle = st.sidebar.slider("Target Flex Angle", 30, 90, 60)
num_sessions = st.sidebar.slider("Number of Sessions", 1, 10, 5)
session_duration = st.sidebar.slider("Session Duration (seconds)", 5, 20, 10)
angle_group = st.sidebar.selectbox("Motion Pattern", [1, 2, 3, 4], index=0)
hand_side = st.sidebar.selectbox("Hand Side", [0, 1], index=0)

start = st.sidebar.button("Start Therapy Session")

# -----------------------------
# Main UI
# -----------------------------
if start:
    results = run_simulation(
        patient_case=severity,
        target_angle=target_angle,
        session_duration=session_duration,
        num_sessions=num_sessions,
        angle_group=angle_group,
        hand_side=hand_side
    )

    # -----------------------------
    # Patient Information
    # -----------------------------
    st.subheader("Patient Information")
    c1, c2, c3, c4 = st.columns(4)
    c1.write(f"**Name:** {patient_name}")
    c2.write(f"**Age:** {age}")
    c3.write(f"**Severity Test Case:** {severity.capitalize()}")
    c4.write(f"**Recommended Rehab Mode:** {results['recommended_mode']}")

    # -----------------------------
    # Recovery Results
    # -----------------------------
    st.subheader("Recovery Results")
    m1, m2, m3 = st.columns(3)
    m4, m5, m6 = st.columns(3)

    m1.metric("Recovery Score", f"{results['recovery_score']}/100")
    m2.metric("Therapy Improvement", f"{results['therapy_improvement']}%")
    m3.metric("Estimated Sessions Left", results["sessions_remaining"])

    m4.metric("Baseline Patient ROM", results["baseline_rom"])
    m5.metric("Actual ROM Progress", results["actual_rom_progress"])
    m6.metric("Assisted ROM Progress", results["rom_progress"])

    # -----------------------------
    # Recovery Trend (PLOTLY VERSION)
    # -----------------------------
    st.subheader("Recovery Trend")

    session_numbers = list(range(1, len(results["sessions"]) + 1))
    recovery_scores = results["sessions"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=session_numbers,
        y=recovery_scores,
        mode="lines+markers",
        name="Recovery Score",
        line=dict(color="#4da3ff", width=3),
        marker=dict(size=9)
    ))

    fig.update_layout(
        title="Recovery Score Improvement Over Sessions",
        xaxis_title="Session Number",
        yaxis_title="Recovery Score",
        template="plotly_dark",
        height=500,
        xaxis=dict(
            tickmode="linear",
            dtick=1
        ),
        yaxis=dict(
            range=[max(0, min(recovery_scores) - 5), min(100, max(recovery_scores) + 5)]
        ),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Meaning of Results
    # -----------------------------
    st.subheader("Meaning of Results")
    st.info(
        "Recovery Score shows the patient's current overall functional capability. "
        "Therapy Improvement shows how much the patient's own range of motion improved across sessions. "
        "Baseline Patient ROM is the initial movement ability without robotic help. "
        "Assisted ROM Progress shows the movement achieved with robotic glove support."
    )

    # -----------------------------
    # Session History
    # -----------------------------
    st.subheader("Session History")
    st.dataframe(pd.DataFrame(results["session_history"]), use_container_width=True)

    # -----------------------------
    # Personalized Recommendation
    # -----------------------------
    st.subheader("Personalized Recommendation")
    st.success(results["recommendation"])

    # -----------------------------
    # Recovery Summary
    # -----------------------------
    st.subheader("Recovery Summary")
    st.write(
        f"Across **{results['num_sessions']} sessions**, the patient's **Recovery Score** reached "
        f"**{results['recovery_score']}/100**."
    )
    st.write(
        f"The patient's **baseline ROM** started at **{results['baseline_rom']}**, "
        f"and their **actual ROM progress** improved to **{results['actual_rom_progress']}**."
    )
    st.write(
        f"With robotic support, the glove enabled an **assisted ROM progression** of "
        f"**{results['rom_progress']}**."
    )
    st.write(
        f"Based on this simulation, approximately **{results['sessions_remaining']} more sessions** "
        f"may be required for stronger recovery."
    )