import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

html, body {
    margin: 0 !important;
    padding: 0 !important;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #07111f, #101a33, #0b1325);
}

[data-testid="stHeader"] {
    height: 0 !important;
    background: transparent !important;
}

.block-container {
    max-width: 100% !important;
    padding-top: 0.5rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
}

/* INPUT BOX */
div[data-baseweb="input"] {
    background-color: white !important;
    border-radius: 10px !important;
}

div[data-baseweb="input"] > div {
    background-color: white !important;
    border: 1px solid #cccccc !important;
}

div[data-baseweb="input"] input {
    color: #111111 !important;
    background-color: white !important;
    -webkit-text-fill-color: #111111 !important;
}

/* SELECT BOX */
div[data-baseweb="select"] > div {
    background-color: white !important;
    border: 1px solid #cccccc !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] input {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
}

div[data-baseweb="select"] span {
    color: #111111 !important;
}

/* NUMBER INPUT */
input[type="number"] {
    color: #111111 !important;
    background-color: white !important;
    -webkit-text-fill-color: #111111 !important;
}

/* LABELS */
label {
    color: white !important;
    font-weight: 600 !important;
}

/* TITLE */
.main-title {
    text-align: center;
    color: white;
    font-size: 3rem;
    font-weight: 800;
}

.subtitle {
    text-align: center;
    color: #c5d0e0;
    margin-bottom: 1.5rem;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    height: 3.2rem;
    border-radius: 12px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white !important;
    font-size: 1.1rem;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

/* Main background */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(89, 70, 170, 0.25), transparent 35%),
        radial-gradient(circle at bottom right, rgba(0, 150, 180, 0.18), transparent 35%),
        linear-gradient(135deg, #07111f 0%, #101a33 50%, #0b1325 100%);
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    color: white;
    margin-top: 0;
    margin-bottom: 0.3rem;
}

.subtitle {
    text-align: center;
    color: #b9c7dd;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}

/* Section heading */
.section-title {
    color: white;
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 0.5rem;
    margin-bottom: 0.3rem;
}

.section-text {
    color: #aebbd0;
    margin-bottom: 1rem;
}

/* Input labels */
label {
    color: #e9eef8 !important;
    font-weight: 600 !important;
}

/* Inputs */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
}

input {
    color: white !important;
}

div[data-baseweb="select"] span {
    color: white !important;
}

/* Cards */
.info-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 1.2rem;
    margin: 0.5rem 0 1rem 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}

/* Button */
.stButton > button {
    width: 100%;
    height: 3.2rem;
    border: none;
    border-radius: 12px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    font-size: 1.1rem;
    font-weight: 700;
    margin-top: 1rem;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(118,75,162,0.4);
}

/* Prediction cards */
.high-risk {
    background: rgba(220, 53, 69, 0.16);
    border: 1px solid rgba(255, 90, 100, 0.45);
    border-radius: 18px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1.5rem;
}

.low-risk {
    background: rgba(25, 180, 110, 0.16);
    border: 1px solid rgba(50, 220, 140, 0.45);
    border-radius: 18px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1.5rem;
}

.result-title {
    color: white;
    font-size: 1.8rem;
    font-weight: 800;
}

.result-probability {
    color: #dce6f7;
    font-size: 1.15rem;
    margin-top: 0.5rem;
}

/* Footer */
.footer {
    text-align: center;
    color: #8291aa;
    font-size: 0.9rem;
    margin-top: 2rem;
    padding-top: 1rem;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================
MODEL_PATH = Path(__file__).resolve().parent /  "best_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("❌ Model file not found. Please check: models/best_model.pkl")
    st.stop()
except Exception as e:
    st.error(f"❌ Model loading error: {e}")
    st.stop()


# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="main-title">👨‍💼 Employee Attrition Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    '🤖 AI-powered system to predict employee attrition risk'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# EMPLOYEE INFORMATION
# =========================================================
st.markdown(
    '<div class="section-title">📋 Employee Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-text">'
    'Enter the employee details below to predict attrition risk.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# ROW 1
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "🎂 Age",
        min_value=18,
        max_value=70,
        value=30
    )

with col2:
    department = st.selectbox(
        "🏢 Department",
        [
            "Sales",
            "Research & Development",
            "Human Resources"
        ]
    )

with col3:
    job_role = st.selectbox(
        "💼 Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )


# =========================================================
# ROW 2
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    monthly_income = st.number_input(
        "💰 Monthly Income",
        min_value=1000,
        max_value=200000,
        value=5000,
        step=500
    )

with col2:
    business_travel = st.selectbox(
        "✈️ Business Travel",
        [
            "Travel_Rarely",
            "Travel_Frequently",
            "Non-Travel"
        ]
    )

with col3:
    distance_from_home = st.number_input(
        "📍 Distance From Home",
        min_value=1,
        max_value=100,
        value=10
    )


# =========================================================
# ROW 3
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    job_satisfaction = st.select_slider(
        "😊 Job Satisfaction",
        options=[1, 2, 3, 4],
        value=3
    )

with col2:
    environment_satisfaction = st.select_slider(
        "🌱 Environment Satisfaction",
        options=[1, 2, 3, 4],
        value=3
    )

with col3:
    job_involvement = st.select_slider(
        "🤝 Job Involvement",
        options=[1, 2, 3, 4],
        value=3
    )


# =========================================================
# ROW 4
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    overtime = st.selectbox(
        "⏰ Overtime",
        ["Yes", "No"]
    )

with col2:
    work_life_balance = st.select_slider(
        "⚖️ Work-Life Balance",
        options=[1, 2, 3, 4],
        value=3
    )

with col3:
    years_at_company = st.number_input(
        "🏆 Years At Company",
        min_value=0,
        max_value=50,
        value=5
    )


# =========================================================
# ROW 5
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    total_working_years = st.number_input(
        "💼 Total Working Years",
        min_value=0,
        max_value=50,
        value=8
    )

with col2:
    companies_worked = st.number_input(
        "🏢 Companies Worked",
        min_value=0,
        max_value=20,
        value=2
    )

with col3:
    marital_status = st.selectbox(
        "💍 Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "🔮 Predict Attrition Risk"
)


# =========================================================
# PREDICTION
# =========================================================
if predict_button:

    input_data = pd.DataFrame([{

        "Age": age,

        "BusinessTravel": business_travel,

        "DailyRate": 750,

        "Department": department,

        "DistanceFromHome": distance_from_home,

        "Education": 3,

        "EducationField": "Life Sciences",

        "EmployeeCount": 1,

        "EmployeeNumber": 9999,

        "EnvironmentSatisfaction": environment_satisfaction,

        "Gender": "Female",

        "HourlyRate": 65,

        "JobInvolvement": job_involvement,

        "JobLevel": 2,

        "JobRole": job_role,

        "JobSatisfaction": job_satisfaction,

        "MaritalStatus": marital_status,

        "MonthlyIncome": monthly_income,

        "MonthlyRate": 15000,

        "NumCompaniesWorked": companies_worked,

        "Over18": "Y",

        "OverTime": overtime,

        "PercentSalaryHike": 15,

        "PerformanceRating": 3,

        "RelationshipSatisfaction": 3,

        "StandardHours": 80,

        "StockOptionLevel": 1,

        "TotalWorkingYears": total_working_years,

        "TrainingTimesLastYear": 3,

        "WorkLifeBalance": work_life_balance,

        "YearsAtCompany": years_at_company,

        "YearsInCurrentRole": 3,

        "YearsSinceLastPromotion": 1,

        "YearsWithCurrManager": 3

    }])

    try:

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1] * 100

        # -------------------------------------------------
        # HIGH RISK
        # -------------------------------------------------
        if prediction == 1:

            st.markdown(
                f"""
                <div class="high-risk">
                    <div class="result-title">
                        🔴 High Attrition Risk
                    </div>
                    <div class="result-probability">
                        Probability of leaving: <b>{probability:.2f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------
        # LOW RISK
        # -------------------------------------------------
        else:

            st.markdown(
                f"""
                <div class="low-risk">
                    <div class="result-title">
                        🟢 Low Attrition Risk
                    </div>
                    <div class="result-probability">
                        Probability of leaving: <b>{probability:.2f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as e:

        st.error(
            f"❌ Prediction error: {e}"
        )


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        🤖 AI-Based Employee Attrition Prediction System
        <br>
        Built with Python • Machine Learning • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
