import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from simulation_engine import run_simulation

st.set_page_config(page_title="AI Robotic Hand Rehabilitation", layout="wide")

# ==============================
# DASHBOARD SWITCH
# ==============================

mode = st.sidebar.selectbox(
    "Select Dashboard",
    ["AI Robotic Hand Rehabilitation", "Therapist Dashboard"],
    key="dashboard_mode"
)

# ==============================
# PATIENT DASHBOARD
# ==============================

if mode == "AI Robotic Hand Rehabilitation":

    st.title("AI Robotic Hand Rehabilitation")

    st.sidebar.header("Patient Profile")

    patient_name = st.sidebar.text_input("Patient Name", "Nidhi", key="patient_name")
    age = st.sidebar.number_input("Age", 18, 90, 25, key="age")

    severity = st.sidebar.selectbox(
        "Severity Level",
        ["severe", "moderate", "mild"],
        key="severity_select"
    )

    st.sidebar.header("Session Setup")

    target_angle = st.sidebar.slider("Target Flex Angle", 30, 90, 60, key="target_angle")
    num_sessions = st.sidebar.slider("Number of Sessions", 1, 10, 5, key="num_sessions")
    session_duration = st.sidebar.slider("Session Duration", 5, 20, 10, key="session_duration")

    angle_group = st.sidebar.selectbox(
        "Motion Pattern",
        [1,2,3,4],
        key="motion_pattern"
    )

    hand_side = st.sidebar.selectbox(
        "Hand Side (0: Left, 1: Right)",
        [0,1],
        key="hand_side"
    )

    start = st.sidebar.button("Start Therapy Session", key="start_button")

    if start:

        results = run_simulation(
            patient_case=severity,
            target_angle=target_angle,
            session_duration=session_duration,
            num_sessions=num_sessions,
            angle_group=angle_group,
            hand_side=hand_side
        )

        # ==============================
        # SAVE DATA FOR THERAPIST DASHBOARD
        # ==============================

        df = pd.DataFrame({
            "PatientName":[patient_name]*len(results["sessions"]),
            "Session":list(range(1,len(results["sessions"])+1)),
            "RecoveryScore":results["sessions"],
            "ROM":results["rom"],
            "GripForce":results["force"],
            "AssistRatio":results["assist"]
        })

        df.to_csv("latest_session.csv",index=False)

        meta_df = pd.DataFrame({
            "Name":[patient_name],
            "Age":[age],
            "Severity":[severity.capitalize()],
            "Hand":["Right" if hand_side==1 else "Left"]
        })

        meta_df.to_csv("patient_meta.csv",index=False)

        # ==============================
        # PATIENT INFO
        # ==============================

        st.subheader("Patient Information")

        c1,c2,c3,c4 = st.columns(4)

        c1.write(f"**Name:** {patient_name}")
        c2.write(f"**Age:** {age}")
        c3.write(f"**Severity:** {severity.capitalize()}")
        c4.write(f"**Recommended Mode:** {results['recommended_mode']}")

        # ==============================
        # RECOVERY METRICS
        # ==============================

        st.subheader("Recovery Results")

        m1,m2,m3 = st.columns(3)

        m1.metric("Recovery Score",f"{results['recovery_score']}/100")
        m2.metric("Therapy Improvement",f"{results['therapy_improvement']}%")
        m3.metric("Estimated Sessions Left",results["sessions_remaining"])

        m4,m5,m6 = st.columns(3)

        m4.metric(
            "Baseline Patient ROM",
            results["baseline_rom"]
        )

        m5.metric(
            "Actual ROM Progress",
            results["actual_rom_progress"]
        )

        m6.metric(
            "Assisted ROM Progress",
            results["rom_progress"]
        )

        # ==============================
        # RECOVERY GRAPH
        # ==============================

        st.subheader("Recovery Trend")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(1,len(results["sessions"])+1)),
            y=results["sessions"],
            mode="lines+markers",
            line=dict(width=3)
        ))

        fig.update_layout(
            title="Recovery Score Improvement Over Sessions",
            xaxis_title="Session",
            yaxis_title="Recovery Score",
            template="plotly_dark"
        )

        st.plotly_chart(fig,use_container_width=True)

        # ==============================
        # MEANING OF RESULTS
        # ==============================

        st.subheader("Meaning of Results")

        st.info(
            "Recovery Score shows the patient's current overall functional capability. "
            "Therapy Improvement shows how much the patient's own range of motion improved across sessions. "
            "Baseline Patient ROM is the initial movement ability without robotic help. "
            "Assisted ROM Progress shows the movement achieved with robotic glove support."
        )

        # ==============================
        # SESSION HISTORY
        # ==============================

        st.subheader("Session History")

        session_df = pd.DataFrame({
            "Session":list(range(1,len(results["sessions"])+1)),
            "Baseline ROM":results["baseline_rom_series"],
            "Assisted ROM":results["rom"],
            "Recovery Score":results["sessions"],
            "Peak Force":results["force"],
            "Assist Ratio":results["assist"]
        })

        st.dataframe(session_df,use_container_width=True)

        # ==============================
        # RECOMMENDATION
        # ==============================

        st.subheader("Personalized Recommendation")

        st.success(results["recommendation"])

        # ==============================
        # SUMMARY
        # ==============================

        st.subheader("Recovery Summary")

        st.write(
            f"Across **{num_sessions} sessions**, the patient's **Recovery Score** reached "
            f"**{results['recovery_score']}/100**."
        )

        st.write(
            f"The patient's **baseline ROM** started at **{results['baseline_rom']}**, "
            f"and their **actual ROM progress** improved to **{results['actual_rom_progress']}**."
        )

        st.write(
            f"With robotic support, the glove enabled an **assisted ROM progression** "
            f"of **{results['rom_progress']}**."
        )

        st.write(
            f"Based on this simulation, approximately **{results['sessions_remaining']} more sessions** "
            f"may be required for stronger recovery."
        )

# ==============================
# THERAPIST DASHBOARD
# ==============================

elif mode == "Therapist Dashboard":

    st.title("Therapist Monitoring Dashboard")

    try:

        df = pd.read_csv("latest_session.csv")
        meta = pd.read_csv("patient_meta.csv").iloc[0]

        st.subheader("Patient Overview")

        c1,c2,c3,c4,c5 = st.columns(5)

        c1.write("**Patient ID:** P-001")
        c2.write(f"**Patient Name:** {meta['Name']}")
        c3.write(f"**Age:** {meta['Age']}")
        c4.write(f"**Stroke Severity:** {meta['Severity']}")
        c5.write(f"**Affected Hand:** {meta['Hand']}")

        st.subheader("Session History")
        st.dataframe(df,use_container_width=True)

        col1,col2 = st.columns(2)

        with col1:

            st.subheader("Recovery Score Trend")

            fig1 = go.Figure()

            fig1.add_trace(go.Scatter(
                x=df["Session"],
                y=df["RecoveryScore"],
                mode="lines+markers"
            ))

            fig1.update_layout(
                xaxis_title="Session",
                yaxis_title="Score",
                template="plotly_dark"
            )

            st.plotly_chart(fig1,use_container_width=True)

        with col2:

            st.subheader("ROM Improvement")

            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(
                x=df["Session"],
                y=df["ROM"],
                mode="lines+markers"
            ))

            fig2.update_layout(
                xaxis_title="Session",
                yaxis_title="ROM (deg)",
                template="plotly_dark"
            )

            st.plotly_chart(fig2,use_container_width=True)

        st.subheader("Grip Strength Trend")

        fig3 = go.Figure()

        fig3.add_trace(go.Scatter(
            x=df["Session"],
            y=df["GripForce"],
            mode="lines+markers"
        ))

        fig3.update_layout(
            xaxis_title="Session",
            yaxis_title="Grip Force (N)",
            template="plotly_dark"
        )

        st.plotly_chart(fig3,use_container_width=True)

        st.divider()

        results = run_simulation(
            patient_case=meta['Severity'].lower(),
            target_angle=60,
            session_duration=10,
            num_sessions=5,
            angle_group=1,
            hand_side=1 if meta['Hand']=="Right" else 0
        )

        st.subheader("Grip Force During Session")

        fig4 = go.Figure()

        fig4.add_trace(go.Scatter(
            x=results["time"],
            y=results["grip_force_curve"],
            mode="lines",
            name="Grip Force"
        ))

        fig4.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Force (N)",
            template="plotly_dark"
        )

        st.plotly_chart(fig4,use_container_width=True)

        st.subheader("Finger Flexion Analysis")

        fig5 = go.Figure()

        for finger in results["finger_curves"]:

            fig5.add_trace(go.Scatter(
                x=results["time"],
                y=results["finger_curves"][finger]["actual"],
                mode="lines",
                name=f"{finger} Actual"
            ))

            fig5.add_trace(go.Scatter(
                x=results["time"],
                y=results["finger_curves"][finger]["assisted"],
                mode="lines",
                name=f"{finger} Assisted",
                line=dict(dash='dash')
            ))

        fig5.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Angle (deg)",
            template="plotly_dark"
        )

        st.plotly_chart(fig5,use_container_width=True)

    except FileNotFoundError:
        st.warning("No session data found. Please run a Patient Therapy Session first.")

    except Exception as e:
        st.error(f"An error occurred: {e}")