import json
import random
import string
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier  # for future training if needed
import joblib


# --------------------------- CONFIG --------------------------------- #

st.set_page_config(
    page_title="Telemetry Based Bot Detection",
    page_icon="🛡️",
    layout="wide",
)

# Feature names used for the (optional) ML model and rules
FEATURE_COLUMNS = [
    "requests_per_min",       # Number of HTTP requests per minute
    "avg_dwell_time",         # Average time spent on a page (seconds)
    "mouse_moves_per_min",    # Mouse movements per minute
    "failed_logins",          # Number of failed login attempts in the session
    "distinct_endpoints"      # Number of different endpoints hit in this session
]


# --------------------------- MODEL LOADING --------------------------- #

@st.cache_resource
def load_ml_model(path: str = "bot_detection_model.pkl"):
    """
    Try to load a trained ML model from disk.
    If not found, return None and the app will use heuristic logic.
    """
    try:
        model = joblib.load(path)
        return model
    except Exception:
        return None


ML_MODEL = load_ml_model()


# --------------------------- RULE ENGINE ----------------------------- #

def rule_based_rating(features: Dict[str, float]) -> Tuple[int, Dict[str, bool]]:
    """
    Very simple rule-based scoring from 1 to 10.
    Higher score = more human / less suspicious.

    Returns:
        rating: int (1-10)
        flags: dict of rule flags explaining why it was suspicious
    """
    score = 10
    flags = {
        "high_request_rate": False,
        "low_dwell_time": False,
        "robotic_mouse": False,
        "many_failed_logins": False,
        "endpoint_scanning": False,
    }

    # Rule 1: Very high request rate -> suspicious
    rpm = features.get("requests_per_min", 0)
    if rpm > 200:
        score -= 6
        flags["high_request_rate"] = True
    elif rpm > 120:
        score -= 4
        flags["high_request_rate"] = True
    elif rpm > 60:
        score -= 2
        flags["high_request_rate"] = True

    # Rule 2: Very low dwell time (fast page flipping)
    dwell = features.get("avg_dwell_time", 0)
    if dwell < 1:
        score -= 4
        flags["low_dwell_time"] = True
    elif dwell < 3:
        score -= 2
        flags["low_dwell_time"] = True

    # Rule 3: Mouse movements – too low can look like script; too high can look like jitter bot
    mouse_rate = features.get("mouse_moves_per_min", 0)
    if mouse_rate < 5:
        score -= 3
        flags["robotic_mouse"] = True
    elif mouse_rate > 300:
        score -= 2
        flags["robotic_mouse"] = True

    # Rule 4: Failed logins
    failed_logins = features.get("failed_logins", 0)
    if failed_logins >= 10:
        score -= 4
        flags["many_failed_logins"] = True
    elif failed_logins >= 5:
        score -= 2
        flags["many_failed_logins"] = True

    # Rule 5: Hitting many endpoints in a short session (probable scanner)
    endpoints = features.get("distinct_endpoints", 0)
    if endpoints > 50:
        score -= 4
        flags["endpoint_scanning"] = True
    elif endpoints > 20:
        score -= 2
        flags["endpoint_scanning"] = True

    # Clamp between 1 and 10
    score = int(max(1, min(10, score)))
    return score, flags


# --------------------------- ML DECISION ----------------------------- #

def ml_prediction(features: Dict[str, float]) -> Tuple[str, float]:
    """
    Predict BOT vs HUMAN using ML model if available.
    If no trained model file, use a fallback heuristic that returns a probability.

    Returns:
        label: "BOT" or "HUMAN"
        prob_bot: probability that it's a bot (0.0 - 1.0)
    """
    # Prepare feature vector
    x = np.array([[features.get(f, 0.0) for f in FEATURE_COLUMNS]])

    if ML_MODEL is not None:
        # Proper sklearn model path (binary classification)
        try:
            proba = ML_MODEL.predict_proba(x)[0][1]  # probability of class "BOT"
        except Exception:
            # Fallback if model has different label ordering
            proba = float(ML_MODEL.predict_proba(x)[0].max())
        label = "BOT" if proba >= 0.5 else "HUMAN"
        return label, float(proba)

    # Heuristic fallback: not real ML, just for demonstration
    rpm = features.get("requests_per_min", 0)
    dwell = features.get("avg_dwell_time", 0)
    mouse = features.get("mouse_moves_per_min", 0)
    failed = features.get("failed_logins", 0)
    endpoints = features.get("distinct_endpoints", 0)

    # Simple scoring
    heuristic_score = 0
    if rpm > 200:
        heuristic_score += 3
    if dwell < 2:
        heuristic_score += 2
    if mouse < 5:
        heuristic_score += 1
    if failed >= 5:
        heuristic_score += 2
    if endpoints > 20:
        heuristic_score += 2

    # Map heuristic_score to a "probability"
    prob_bot = min(1.0, heuristic_score / 7.0)
    label = "BOT" if prob_bot >= 0.5 else "HUMAN"
    return label, prob_bot


# --------------------------- DECISION ENGINE ------------------------- #

def decision_engine(
    rating: int,
    rule_flags: Dict[str, bool],
    ml_label: str,
    ml_prob_bot: float
) -> Tuple[str, List[str]]:
    """
    Combine Rules + ML result to decide what to do.

    Returns:
        decision: "ALLOW", "CAPTCHA_REQUIRED", or "BLOCK"
        reasons: list of text reasons
    """
    reasons = []

    # Pure ML very strong signal
    if ml_label == "BOT" and ml_prob_bot >= 0.9:
        reasons.append("ML model is highly confident this is a bot.")
        return "CAPTCHA_REQUIRED", reasons

    # Rating-based logic
    if rating >= 8:
        # High rating → likely human
        reasons.append("Rule-based rating is high (8–10).")
        if ml_label == "BOT":
            reasons.append("ML is slightly suspicious, but rules indicate human-like behaviour.")
            # Still allow but log – for now we just allow
        return "ALLOW", reasons

    if rating < 5:
        reasons.append("Rule-based rating is low (<5). Highly suspicious.")
        # Low rating → definitely suspicious; require captcha
        return "CAPTCHA_REQUIRED", reasons

    # Rating between 5 and 7 → depends on flags and ML
    suspicious_flags = [k for k, v in rule_flags.items() if v]
    if suspicious_flags:
        reasons.append(
            f"Medium rating (5–7) with suspicious rule flags: {', '.join(suspicious_flags)}"
        )

    if ml_label == "BOT" and ml_prob_bot >= 0.6:
        reasons.append("ML model also flags as bot with prob >= 0.6.")
        return "CAPTCHA_REQUIRED", reasons

    if suspicious_flags:
        # rules say suspicious but ML is not confident -> still trigger captcha
        return "CAPTCHA_REQUIRED", reasons

    # Otherwise allow
    reasons.append("Medium rating but no strong suspicious patterns detected.")
    return "ALLOW", reasons


# --------------------------- CAPTCHA LOGIC --------------------------- #

def generate_captcha_text(length: int = 5) -> str:
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=length))


def init_captcha():
    if "captcha_text" not in st.session_state:
        st.session_state["captcha_text"] = generate_captcha_text()
    if "captcha_status" not in st.session_state:
        st.session_state["captcha_status"] = None  # None / "PASSED" / "FAILED"


def render_captcha_section():
    st.subheader("CAPTCHA Verification 🔐")

    init_captcha()

    st.write("Please enter the characters shown below to prove you are a human.")
    st.markdown(
        f"""
        <div style="
            font-size: 32px;
            font-weight: bold;
            letter-spacing: 8px;
            padding: 10px 20px;
            border: 2px dashed #888;
            display: inline-block;
            margin-bottom: 10px;
        ">
            {st.session_state['captcha_text']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_input = st.text_input("Enter CAPTCHA text exactly as shown:", key="captcha_input")
    if st.button("Submit CAPTCHA"):
        if user_input.strip().upper() == st.session_state["captcha_text"]:
            st.success("✅ CAPTCHA correct. Access allowed. Detected as HUMAN.")
            st.session_state["captcha_status"] = "PASSED"
        else:
            st.error("❌ CAPTCHA failed. Session/IP should be BLOCKED.")
            st.session_state["captcha_status"] = "FAILED"

        # regenerate captcha for next time
        st.session_state["captcha_text"] = generate_captcha_text()


# --------------------------- TELEMETRY INPUT UI ---------------------- #

def parse_json_telemetry(raw: str) -> Dict[str, float]:
    """
    Parse JSON telemetry string safely.
    """
    data = json.loads(raw)
    features = {}
    for col in FEATURE_COLUMNS:
        features[col] = float(data.get(col, 0.0))
    return features


def form_telemetry_input() -> Dict[str, float]:
    col1, col2, col3 = st.columns(3)

    with col1:
        requests_per_min = st.number_input(
            "Requests per minute",
            min_value=0.0,
            max_value=1000.0,
            value=30.0,
            step=5.0,
        )
        failed_logins = st.number_input(
            "Failed logins in this session",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
        )

    with col2:
        avg_dwell_time = st.number_input(
            "Average dwell time (seconds)",
            min_value=0.0,
            max_value=600.0,
            value=8.0,
            step=0.5,
        )
        distinct_endpoints = st.number_input(
            "Distinct endpoints accessed",
            min_value=0.0,
            max_value=200.0,
            value=5.0,
            step=1.0,
        )

    with col3:
        mouse_moves_per_min = st.number_input(
            "Mouse moves per minute",
            min_value=0.0,
            max_value=1000.0,
            value=40.0,
            step=5.0,
        )

    return {
        "requests_per_min": requests_per_min,
        "avg_dwell_time": avg_dwell_time,
        "mouse_moves_per_min": mouse_moves_per_min,
        "failed_logins": failed_logins,
        "distinct_endpoints": distinct_endpoints,
    }


# --------------------------- STATEFUL DETECTION ---------------------- #

def run_detection_and_store(telemetry_features: Dict[str, float]):
    """
    Run rules + ML + decision and store everything in session_state.
    This makes the app survive Streamlit re-runs (e.g. when pressing buttons).
    """
    rating, rule_flags = rule_based_rating(telemetry_features)
    ml_label, ml_prob_bot = ml_prediction(telemetry_features)
    decision, reasons = decision_engine(rating, rule_flags, ml_label, ml_prob_bot)

    st.session_state["has_detection"] = True
    st.session_state["telemetry_features"] = telemetry_features
    st.session_state["rating"] = rating
    st.session_state["rule_flags"] = rule_flags
    st.session_state["ml_label"] = ml_label
    st.session_state["ml_prob_bot"] = ml_prob_bot
    st.session_state["decision"] = decision
    st.session_state["reasons"] = reasons

    # Reset CAPTCHA state for new detection
    st.session_state["captcha_status"] = None
    st.session_state["captcha_text"] = generate_captcha_text()


# --------------------------- MAIN APP -------------------------------- #

def main():
    # Initialise global flags in session_state
    if "has_detection" not in st.session_state:
        st.session_state["has_detection"] = False

    st.title("🛡️ Telemetry Based Bot Detection (Rules + ML + CAPTCHA)")
    st.caption("Cyber Security Major Project – Python 3.10 + Streamlit backend")

    with st.sidebar:
        st.header("How this demo works")
        st.write(
            """
            1. **Telemetry data** from the browser (simulated here) is sent as JSON.\n
            2. A **Rule Engine** gives a rating from 1–10.\n
            3. An optional **ML model** (if `bot_detection_model.pkl` exists) predicts BOT/HUMAN.\n
            4. The **Decision Engine** combines both.\n
               - If clearly human → **Allow Access**.\n
               - If suspicious → **Trigger CAPTCHA**.\n
               - If CAPTCHA fails → **Block Session/IP**.
            """
        )
        st.markdown("---")
        st.write("You can later replace the heuristic model with your own trained model.")

    st.subheader("Step 1 – Provide Telemetry Data")

    tab_form, tab_json = st.tabs(["Form Input (easy)", "Raw JSON Input"])

    # --------- INPUT SECTION (only triggers detection when buttons clicked) --------- #
    with tab_form:
        st.write("Fill in the telemetry values to simulate a user/session.")
        telemetry_features_form = form_telemetry_input()
        if st.button("Run Detection (Form)", key="run_form"):
            run_detection_and_store(telemetry_features_form)

    with tab_json:
        sample_json = json.dumps(
            {
                "requests_per_min": 25,
                "avg_dwell_time": 12,
                "mouse_moves_per_min": 80,
                "failed_logins": 0,
                "distinct_endpoints": 6,
            },
            indent=2,
        )
        raw_json = st.text_area(
            "Paste telemetry JSON here",
            value=sample_json,
            height=180,
        )
        if st.button("Run Detection (JSON)", key="run_json"):
            try:
                telemetry_features_json = parse_json_telemetry(raw_json)
                run_detection_and_store(telemetry_features_json)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    # If we still don't have any detection, stop here
    if not st.session_state["has_detection"]:
        st.info("Submit telemetry using either **Form** or **Raw JSON** to run detection.")
        return

    # --------- DISPLAY STORED RESULTS (this survives reruns) --------- #
    telemetry_features = st.session_state["telemetry_features"]
    rating = st.session_state["rating"]
    rule_flags = st.session_state["rule_flags"]
    ml_label = st.session_state["ml_label"]
    ml_prob_bot = st.session_state["ml_prob_bot"]
    decision = st.session_state["decision"]
    reasons = st.session_state["reasons"]

    st.subheader("Step 2 – Rule Based Classification (Rating)")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.metric("Rule-based Rating (1–10)", value=rating)
    with col_r2:
        suspicious_flags = [k for k, v in rule_flags.items() if v]
        if suspicious_flags:
            st.warning("Suspicious rule flags: " + ", ".join(suspicious_flags))
        else:
            st.success("No strong suspicious patterns detected from rules.")

    st.subheader("Step 3 – ML Model Prediction (Optional)")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("ML Prediction Label", value=ml_label)
    with col_m2:
        st.metric("Probability of BOT (0–1)", value=f"{ml_prob_bot:.2f}")

    st.caption(
        "If you put a trained model as `bot_detection_model.pkl` in the project folder, "
        "this section will use it automatically."
    )

    st.subheader("Step 4 – Decision Engine")

    if decision == "ALLOW":
        st.success("✅ Final Decision: ALLOW ACCESS (Human).")
    elif decision == "CAPTCHA_REQUIRED":
        st.warning("⚠️ Final Decision: CAPTCHA REQUIRED (Suspicious Session).")
    else:
        st.error("⛔ Final Decision: BLOCK (Very high risk bot).")

    st.write("**Explanation / Reasons:**")
    if reasons:
        for r in reasons:
            st.markdown(f"- {r}")
    else:
        st.markdown("- No additional reasons (default behaviour).")

    # --------- CAPTCHA SECTION (now survives reruns) --------- #
    if decision == "CAPTCHA_REQUIRED":
        st.markdown("---")
        render_captcha_section()

        status = st.session_state.get("captcha_status")
        if status == "PASSED":
            st.success("✅ Access finally ALLOWED after successful CAPTCHA.")
        elif status == "FAILED":
            st.error("⛔ Access BLOCKED – CAPTCHA failed, treat as BOT.")
        else:
            st.info("Waiting for user to complete CAPTCHA verification.")


if __name__ == "__main__":
    main()
