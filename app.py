import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# ============================================================
# LOAD MODEL & SCALER
# ============================================================
MODEL_PATH = r"C:\Users\HP\Desktop\FuWukari\Eating Disorder\ed_risk_model.h5"
SCALER_PATH = r"C:\Users\HP\Desktop\FuWukari\Eating Disorder\ed_scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(
    page_title="Eating Disorder Early Detection AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS FOR BEAUTIFUL UI
# ============================================================
st.markdown("""
<style>

body {
    background: #F8FAFC;
}

.login-box {
    background: white;
    padding: 40px;
    border-radius: 12px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    width: 380px;
    margin: auto;
}

.input-card {
    background: white;
    padding: 35px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.10);
}

.result-card {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.10);
}

.badge-low {
    background-color: #16A34A;
    color: white;
    padding: 6px 12px;
    border-radius: 8px;
    font-weight: 600;
}

.badge-moderate {
    background-color: #F59E0B;
    color: white;
    padding: 6px 12px;
    border-radius: 8px;
    font-weight: 600;
}

.badge-high {
    background-color: #DC2626;
    color: white;
    padding: 6px 12px;
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOGIN PAGE
# ============================================================
def login_screen():
    st.markdown("<h2 style='text-align:center;'>🔐 Login to Access the ED Risk System</h2>", unsafe_allow_html=True)
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="Enter username")
    password = st.text_input("Password", type="password", placeholder="Enter password")

    login_btn = st.button("Login")

    st.markdown("</div>", unsafe_allow_html=True)

    if login_btn:
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password.")

# ============================================================
# CLINICAL INTERPRETATION
# ============================================================
def interpret_risk(prob):
    if prob < 0.33:
        return "Low Risk", "badge-low"
    elif 0.33 <= prob < 0.66:
        return "Moderate Risk", "badge-moderate"
    else:
        return "High Risk", "badge-high"

# ============================================================
# PREDICTION PAGE
# ============================================================
def prediction_screen():
    st.markdown("<h2 style='text-align:center;'>🧠 Eating Disorder Early Risk Prediction</h2>", unsafe_allow_html=True)
    st.write("Enter the required information to assess the likelihood of an eating disorder risk.")

    st.markdown("<div class='input-card'>", unsafe_allow_html=True)

    # Input fields
    r1 = st.number_input("Restrictive Eating Score", 0.0, 3.0, step=0.1)
    r2 = st.number_input("Emotional Eating Score", 0.0, 3.0, step=0.1)
    r3 = st.number_input("Reward Sensitivity Score", 0.0, 3.0, step=0.1)
    r4 = st.number_input("Habitual Eating Score", 0.0, 3.0, step=0.1)
    r5 = st.number_input("Body Image Distress Score", 0.0, 3.0, step=0.1)
    r6 = st.number_input("Overall ED Risk Score", 0.0, 3.0, step=0.1)
    r7 = st.number_input("Demographic Score (encoded)", 0.0, 3.0, step=0.1)

    st.markdown("</div>", unsafe_allow_html=True)

    submit = st.button("🔎 Predict Risk")

    if submit:
        try:
            X = np.array([[r1, r2, r3, r4, r5, r6, r7]])
            X_scaled = scaler.transform(X)

            prob = float(model.predict(X_scaled)[0])
            label, badge_class = interpret_risk(prob)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
            st.markdown(f"<h3>Prediction Result</h3>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Risk Level:</strong> <span class='{badge_class}'>{label}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Confidence Score:</strong> {prob:.4f}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error occurred: {e}")

# ============================================================
# ROUTING
# ============================================================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_screen()
else:
    prediction_screen()
