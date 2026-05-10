import numpy as np
import pandas as pd

np.random.seed(42)

PATIENT_PROFILES = {
    "severe": {
        "severity_scale": 0.28,
        "base_grip_force": 4,
        "fatigue_factor": 0.20,
        "base_assist_ratio": 0.95,
        "growth_rate": 0.05
    },
    "moderate": {
        "severity_scale": 0.55,
        "base_grip_force": 8,
        "fatigue_factor": 0.12,
        "base_assist_ratio": 0.75,
        "growth_rate": 0.06
    },
    "mild": {
        "severity_scale": 0.80,
        "base_grip_force": 12,
        "fatigue_factor": 0.07,
        "base_assist_ratio": 0.45,
        "growth_rate": 0.04
    }
}

FINGERS = {
    "Thumb": 0.90,
    "Index": 1.00,
    "Middle": 1.05,
    "Ring": 0.95,
    "Little": 0.85
}


def resize_to_length(arr, length):
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, length)
    return np.interp(x_new, x_old, arr)


def normalize_motion(arr, max_angle):
    arr = arr - np.min(arr)
    arr_range = np.max(arr) - np.min(arr)
    if arr_range == 0:
        return np.zeros_like(arr)
    arr = arr / arr_range
    return arr * max_angle


def create_target_curve(time_array, session_duration, target):
    curve = np.zeros_like(time_array)

    t1 = 0.2 * session_duration
    t2 = 0.4 * session_duration
    t3 = 0.6 * session_duration
    t4 = 0.8 * session_duration

    for i, t in enumerate(time_array):
        if t1 <= t < t2:
            curve[i] = (t - t1) / (t2 - t1) * target
        elif t2 <= t < t3:
            curve[i] = target
        elif t3 <= t < t4:
            curve[i] = (1 - (t - t3) / (t4 - t3)) * target
        else:
            curve[i] = 0

    return curve


def moving_average(arr, window=5):
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def adaptive_assistance(previous_recovery_score, base_assist_ratio):
    if previous_recovery_score < 40:
        return min(0.98, base_assist_ratio + 0.08)
    elif previous_recovery_score < 65:
        return base_assist_ratio
    else:
        return max(0.25, base_assist_ratio - 0.12)


def recommended_mode(patient_case):
    if patient_case == "severe":
        return "Passive"
    elif patient_case == "moderate":
        return "Active Assisted"
    return "Resistance"


def estimate_sessions_remaining(recovery_score):
    if recovery_score >= 90:
        return 1
    elif recovery_score >= 80:
        return 2
    elif recovery_score >= 70:
        return 4
    elif recovery_score >= 60:
        return 6
    elif recovery_score >= 50:
        return 8
    else:
        return 10


def personalized_recommendation(patient_case, recovery_score, therapy_improvement):
    mode = recommended_mode(patient_case)

    if recovery_score >= 85:
        return f"Continue {mode.lower()} therapy with reduced support. Functional recovery is strong."
    elif therapy_improvement >= 15:
        return f"Good progress observed. Continue {mode.lower()} therapy and gradually increase task difficulty."
    elif therapy_improvement >= 5:
        return f"Moderate improvement observed. Maintain the current {mode.lower()} therapy plan."
    else:
        return f"Improvement is still limited. Continue supervised {mode.lower()} therapy with close monitoring."


def predict_future_score(session_numbers, scores, future_session):
    x = np.array(session_numbers)
    y = np.array(scores)

    if len(x) < 2:
        return float(y[-1])

    coeffs = np.polyfit(x, y, 1)
    predicted = coeffs[0] * future_session + coeffs[1]
    return float(np.clip(predicted, 0, 100))


def run_simulation(
    patient_case,
    target_angle,
    session_duration,
    num_sessions,
    angle_group=1,
    hand_side=0,
    num_points=300
):
    target_angle = float(np.clip(target_angle, 30, 90))
    session_duration = float(np.clip(session_duration, 5, 20))
    num_sessions = int(np.clip(num_sessions, 1, 12))

    profile = PATIENT_PROFILES[patient_case]

    # DATASET-BASED MOTION SOURCE
    data = pd.read_csv("data.csv")
    filtered = data[
        (data["RightorLeft"] == hand_side) &
        (data["Angle number"] == angle_group)
    ].copy()

    if len(filtered) < 20:
        filtered = data.copy()

    raw_motion = filtered["Angle"].values
    base_motion_curve = resize_to_length(raw_motion, num_points)
    base_motion_curve = normalize_motion(base_motion_curve, target_angle)

    time = np.linspace(0, session_duration, num_points)
    dt = time[1] - time[0]
    target_curve = create_target_curve(time, session_duration, target_angle)

    all_session_scores = []
    all_assisted_rom = []
    all_actual_rom = []
    all_session_force = []
    all_assist_ratios = []
    session_history = []

    last_session_data = None
    last_total_grip_force = None
    previous_recovery_score = None

    for session in range(1, num_sessions + 1):
        improvement_factor = 1 + min((session - 1) * profile["growth_rate"], 0.30)

        if previous_recovery_score is None:
            previous_recovery_score = {
                "severe": 25,
                "moderate": 45,
                "mild": 65
            }[patient_case]

        assist_ratio = adaptive_assistance(
            previous_recovery_score,
            profile["base_assist_ratio"]
        )

        session_data = {}
        finger_peak_actual = []
        finger_peak_assisted = []
        total_grip_force = np.zeros_like(time)

        t_start = 0.2 * session_duration
        t_end = 0.8 * session_duration
        t_hold_start = 0.4 * session_duration
        t_hold_end = 0.6 * session_duration

        for finger_name, finger_scale in FINGERS.items():
            # ACTUAL patient motion from dataset + severity scaling
            actual_curve = (
                base_motion_curve *
                profile["severity_scale"] *
                finger_scale *
                improvement_factor
            )

            actual_curve = np.minimum(actual_curve, target_curve)

            fatigue_multiplier = np.ones_like(time)
            for i, t in enumerate(time):
                if t_start <= t < t_end:
                    active_progress = (t - t_start) / (t_end - t_start)
                    fatigue_multiplier[i] = 1 - profile["fatigue_factor"] * active_progress

            actual_curve = actual_curve * fatigue_multiplier
            actual_curve += np.random.normal(
                0,
                {"severe": 0.12, "moderate": 0.08, "mild": 0.05}[patient_case],
                size=num_points
            )
            actual_curve = np.clip(actual_curve, 0, None)

            # ASSISTED motion
            assist_curve = np.zeros_like(time)
            assisted_curve = actual_curve.copy()

            for i, t in enumerate(time):
                if t_start <= t < t_end:
                    gap = target_curve[i] - actual_curve[i]
                    if gap > 0:
                        assist_curve[i] = assist_ratio * gap
                        assisted_curve[i] = actual_curve[i] + assist_curve[i]

            assisted_curve = np.clip(assisted_curve, 0, target_angle)

            # force proxy from assisted motion
            finger_force = (
                (assisted_curve / max(target_angle, 1)) *
                (profile["base_grip_force"] * finger_scale * improvement_factor)
            )

            for i, t in enumerate(time):
                if t_hold_start <= t < t_hold_end:
                    hold_progress = (t - t_hold_start) / (t_hold_end - t_hold_start)
                    finger_force[i] *= (1 - 0.05 * hold_progress)

            finger_force += np.random.normal(0, 0.04, size=num_points)
            finger_force = np.clip(finger_force, 0, None)

            total_grip_force += finger_force
            finger_peak_actual.append(np.max(actual_curve))
            finger_peak_assisted.append(np.max(assisted_curve))

            session_data[finger_name] = {
                "actual_curve": actual_curve,
                "assisted_curve": assisted_curve,
                "force_curve": finger_force
            }

        avg_actual_rom = float(min(np.mean(finger_peak_actual), target_angle))
        avg_assisted_rom = float(min(np.mean(finger_peak_assisted), target_angle))

        # non-decreasing patient progress
        if len(all_actual_rom) > 0:
            avg_actual_rom = max(avg_actual_rom, all_actual_rom[-1] + 0.2)
            avg_actual_rom = min(avg_actual_rom, target_angle)

        if len(all_assisted_rom) > 0:
            avg_assisted_rom = max(avg_assisted_rom, all_assisted_rom[-1] + 0.2)
            avg_assisted_rom = min(avg_assisted_rom, target_angle)

        peak_grip_force = float(np.max(total_grip_force))
        if len(all_session_force) > 0:
            peak_grip_force = max(peak_grip_force, all_session_force[-1] + 0.3)

        success_rate = float(min(100, (avg_assisted_rom / target_angle) * 100))

        endurance_threshold = 0.8 * peak_grip_force
        endurance_time = float(np.sum(total_grip_force >= endurance_threshold) * dt)

        hold_mask = (time >= t_hold_start) & (time < t_hold_end)
        hold_force = total_grip_force[hold_mask]
        if len(hold_force) > 1 and hold_force[0] > 0:
            fatigue_index = float(max(0, (hold_force[0] - hold_force[-1]) / hold_force[0]) * 100)
        else:
            fatigue_index = 0.0

        # IMPORTANT FIX:
        # Recovery score is now based mainly on ACTUAL patient ability, not assisted motion
        rom_score = min((avg_actual_rom / target_angle) * 100, 100)
        force_score = min((peak_grip_force / 60) * 100, 100)
        endurance_score = min((endurance_time / (0.22 * session_duration)) * 100, 100)

        recovery_score = (
            0.50 * rom_score +
            0.20 * force_score +
            0.15 * endurance_score +
            0.15 * success_rate
        )
        recovery_score = float(min(100, recovery_score))

        if len(all_session_scores) > 0:
            recovery_score = max(recovery_score, all_session_scores[-1] + 0.4)
            recovery_score = min(recovery_score, 100)

        all_actual_rom.append(round(avg_actual_rom, 2))
        all_assisted_rom.append(round(avg_assisted_rom, 2))
        all_session_scores.append(round(recovery_score, 2))
        all_session_force.append(round(peak_grip_force, 2))
        all_assist_ratios.append(round(assist_ratio, 2))

        session_history.append({
            "Session": session,
            "Baseline ROM": round(avg_actual_rom, 2),
            "Assisted ROM": round(avg_assisted_rom, 2),
            "Recovery Score": round(recovery_score, 2),
            "Peak Force": round(peak_grip_force, 2),
            "Assist Ratio": round(assist_ratio, 2)
        })

        previous_recovery_score = recovery_score
        last_session_data = session_data
        last_total_grip_force = total_grip_force

    initial_actual_rom = all_actual_rom[0]
    final_actual_rom = all_actual_rom[-1]
    therapy_improvement = round(((final_actual_rom - initial_actual_rom) / target_angle) * 100, 2)
    therapy_improvement = max(0, therapy_improvement)

    baseline_rom_text = f"{initial_actual_rom:.2f}°"
    actual_rom_progress = f"{initial_actual_rom:.2f}° → {final_actual_rom:.2f}°"
    assisted_rom_progress = f"{all_assisted_rom[0]:.2f}° → {all_assisted_rom[-1]:.2f}°"

    predicted_score = round(
        predict_future_score(
            list(range(1, num_sessions + 1)),
            all_session_scores,
            num_sessions + 3
        ),
        2
    )

    sessions_remaining = estimate_sessions_remaining(all_session_scores[-1])
    recommendation = personalized_recommendation(
        patient_case,
        all_session_scores[-1],
        therapy_improvement
    )

    display_session_data = {}
    for finger_name in FINGERS.keys():
        display_session_data[finger_name] = {
            "actual_curve": moving_average(last_session_data[finger_name]["actual_curve"], window=5),
            "assisted_curve": moving_average(last_session_data[finger_name]["assisted_curve"], window=5)
        }

    display_total_grip_force = moving_average(last_total_grip_force, window=5)

    return {
        "patient_case": patient_case,
        "recommended_mode": recommended_mode(patient_case),
        "target_angle": target_angle,
        "session_duration": session_duration,
        "num_sessions": num_sessions,
        "recovery_score": all_session_scores[-1],
        "therapy_improvement": therapy_improvement,
        "baseline_rom": baseline_rom_text,
        "actual_rom_progress": actual_rom_progress,
        "rom_progress": assisted_rom_progress,
        "predicted_score": predicted_score,
        "sessions_remaining": sessions_remaining,
        "recommendation": recommendation,
        "sessions": all_session_scores,
        "rom": all_assisted_rom,
        "baseline_rom_series": all_actual_rom,
        "assist": all_assist_ratios,
        "force": all_session_force,
        "session_history": session_history,
        "time": time.tolist(),
        "target_curve": target_curve.tolist(),
        "grip_force_curve": display_total_grip_force.tolist(),
        "finger_curves": {
            finger: {
                "actual": display_session_data[finger]["actual_curve"].tolist(),
                "assisted": display_session_data[finger]["assisted_curve"].tolist()
            }
            for finger in FINGERS.keys()
        }
    }