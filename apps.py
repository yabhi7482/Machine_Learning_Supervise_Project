
import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #d62828;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555555;
    margin-bottom: 30px;
}

.section-title {
    font-size: 25px;
    font-weight: bold;
    color: #333333;
    margin-top: 20px;
    margin-bottom: 15px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("logistic_regression_model.pkl")


try:
    clf = load_model()
except Exception as e:
    st.error("❌ Model load nahi hua.")
    st.write(e)
    st.stop()


# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="main-title">❤️ Heart Disease Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Heart Disease Prediction System</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# INPUT TITLE
# =========================================================
st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)


# =========================================================
# INPUTS
# =========================================================

col1, col2, col3 = st.columns(3)


# ---------------- COLUMN 1 ----------------

with col1:

    st.subheader("🧑 Basic Information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )


# ---------------- COLUMN 2 ----------------

with col2:

    st.subheader("🩸 Blood Information")

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=700,
        value=200
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1]
    )

    max_hr = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )


# ---------------- COLUMN 3 ----------------

with col3:

    st.subheader("❤️ Heart Information")

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["N", "Y"]
    )

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=-5.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )


st.divider()


# =========================================================
# PREDICTION INTERFACE
# =========================================================

st.markdown(
    '<div class="section-title">🔮 Prediction</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the patient's information above and click the button below."
)


# Center button
col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    predict = st.button(
        "🔍 Predict Heart Disease",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if predict:

    # Create dataframe
    new_patient = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }])


    # =====================================================
    # ONE HOT ENCODING
    # =====================================================

    new_patient_encoded = pd.get_dummies(
        new_patient,
        columns=[
            "Sex",
            "ChestPainType",
            "RestingECG",
            "ExerciseAngina",
            "ST_Slope"
        ],
        drop_first=True,
        dtype=int
    )


    # =====================================================
    # ALIGN MODEL COLUMNS
    # =====================================================

    try:

        model_columns = clf.feature_names_in_

        new_patient_ready = new_patient_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )

    except Exception as e:

        st.error("❌ Model columns match nahi ho rahe.")
        st.write(e)
        st.stop()


    # =====================================================
    # PREDICTION
    # =====================================================

    prediction = clf.predict(new_patient_ready)[0]

    probability = clf.predict_proba(
        new_patient_ready
    )[0][1]


    # =====================================================
    # RESULT INTERFACE
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )


    if prediction == 1:

        st.error(
            "⚠️ HEART DISEASE RISK DETECTED"
        )

        st.metric(
            "Heart Disease Probability",
            f"{probability * 100:.2f}%"
        )

        st.warning(
            "The model predicts a possibility of heart disease. "
            "Please consult a qualified healthcare professional."
        )

    else:

        st.success(
            "✅ NO HEART DISEASE DETECTED"
        )

        st.metric(
            "Heart Disease Probability",
            f"{probability * 100:.2f}%"
        )

        st.info(
            "The model predicts a lower possibility of heart disease."
        )


    # =====================================================
    # PROBABILITY
    # =====================================================

    st.subheader("📈 Prediction Probability")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "No Heart Disease",
            f"{(1 - probability) * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Heart Disease",
            f"{probability * 100:.2f}%"
        )

    st.progress(float(probability))


    # =====================================================
    # PATIENT SUMMARY
    # =====================================================

    st.subheader("📋 Patient Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Age",
            "Sex",
            "Chest Pain Type",
            "Resting BP",
            "Cholesterol",
            "Fasting Blood Sugar",
            "Resting ECG",
            "Maximum Heart Rate",
            "Exercise Angina",
            "Oldpeak",
            "ST Slope"
        ],

        "Value": [
            age,
            sex,
            chest_pain,
            resting_bp,
            cholesterol,
            fasting_bs,
            resting_ecg,
            max_hr,
            exercise_angina,
            oldpeak,
            st_slope
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "⚠️ This application is for educational purposes only and "
    "should not be used as a medical diagnosis."
)
