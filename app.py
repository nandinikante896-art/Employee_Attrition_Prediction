import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0, 150, 255, 0.20), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(150, 80, 255, 0.20), transparent 25%),
        linear-gradient(135deg, #07111f 0%, #101c35 45%, #172b4d 100%);
    color: white;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* Hero section */
.hero {
    padding: 35px;
    border-radius: 25px;
    background:
        linear-gradient(
            135deg,
            rgba(15, 76, 129, 0.95),
            rgba(89, 45, 145, 0.92)
        );
    box-shadow: 0 15px 45px rgba(0,0,0,0.35);
    margin-bottom: 25px;
    border: 1px solid rgba(255,255,255,0.15);
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
    color: white;
}

.hero p {
    font-size: 18px;
    color: #e8efff;
}

/* Employee icons */
.employee-icons {
    text-align: center;
    font-size: 45px;
    letter-spacing: 15px;
    margin-top: 18px;
}

/* Section cards */
.section-card {
    background: rgba(255,255,255,0.08);
    padding: 22px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
}

/* Input labels */
label {
    color: white !important;
    font-weight: 600 !important;
}

/* Button */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    height: 55px;
    font-size: 18px;
    font-weight: 700;
    background: linear-gradient(90deg, #00c6ff, #7b2cff);
    color: white;
    border: none;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.4);
}

/* Result cards */
.success-card {
    background: linear-gradient(
        135deg,
        rgba(0, 180, 120, 0.25),
        rgba(0, 255, 170, 0.10)
    );
    border: 1px solid rgba(0,255,170,0.4);
    padding: 30px;
    border-radius: 22px;
    text-align: center;
}

.danger-card {
    background: linear-gradient(
        135deg,
        rgba(255, 60, 80, 0.25),
        rgba(255, 120, 50, 0.10)
    );
    border: 1px solid rgba(255,80,80,0.4);
    padding: 30px;
    border-radius: 22px;
    text-align: center;
}

.result-title {
    font-size: 32px;
    font-weight: 800;
}

.probability {
    font-size: 26px;
    font-weight: 700;
    margin-top: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: #aebbd2;
    margin-top: 35px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================
MODEL_PATH = Path(__file__).resolve().parent / "best_model.pkl"

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">

<h1>👨‍💼 Employee Attrition Prediction</h1>

<p>
AI-powered system that predicts whether an employee has
<strong>Low</strong> or <strong>High</strong> attrition risk.
</p>

<div class="employee-icons">
👩‍💼 👨‍💻 👩‍💻 👨‍💼 🧑‍💻
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# INPUT SECTION
# =========================================================
st.markdown("""
<div class="section-card">
<h2>📋 Employee Information</h2>
<p>Enter the employee details below.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "🎂 Age",
        min_value=18,
        max_value=65,
        value=30
    )

    department = st.selectbox(
        "🏢 Department",
        [
            "Sales",
            "Research & Development",
            "Human Resources"
        ]
    )

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

    monthly_income = st.number_input(
        "💰 Monthly Income",
        min_value=1000,
        max_value=50000,
        value=5000
    )

    business_travel = st.selectbox(
        "✈️ Business Travel",
        [
            "Travel_Rarely",
            "Travel_Frequently",
            "Non-Travel"
        ]
    )


with col2:

    distance_from_home = st.number_input(
        "📍 Distance From Home",
        min_value=1,
        max_value=30,
        value=5
    )

    job_satisfaction = st.slider(
        "😊 Job Satisfaction",
        1, 4, 3
    )

    environment_satisfaction = st.slider(
        "🌱 Environment Satisfaction",
        1, 4, 3
    )

    job_involvement = st.slider(
        "🤝 Job Involvement",
        1, 4, 3
    )

    overtime = st.selectbox(
        "⏰ Overtime",
        ["Yes", "No"]
    )


with col3:

    work_life_balance = st.slider(
        "⚖️ Work-Life Balance",
        1, 4, 3
    )

    years_at_company = st.number_input(
        "🏆 Years At Company",
        min_value=0,
        max_value=40,
        value=5
    )

    total_working_years = st.number_input(
        "📅 Total Working Years",
        min_value=0,
        max_value=45,
        value=8
    )

    companies_worked = st.number_input(
        "🏭 Companies Worked",
        min_value=0,
        max_value=10,
        value=2
    )

    marital_status = st.selectbox(
        "💍 Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )


# =========================================================
# PREDICTION
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Predict Employee Attrition Risk"):

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

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1] * 100

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # RESULT
    # =====================================================

    if prediction == 1:

        st.markdown(f"""
        <div class="danger-card">

        <div class="result-title">
        🔴 High Attrition Risk
        </div>

        <div class="probability">
        Risk Probability: {probability:.2f}%
        </div>

        <p>
        This employee shows a higher predicted likelihood
        of leaving the organization.
        </p>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="success-card">

        <div class="result-title">
        🟢 Low Attrition Risk
        </div>

        <div class="probability">
        Risk Probability: {probability:.2f}%
        </div>

        <p>
        This employee shows a lower predicted likelihood
        of leaving the organization.
        </p>

        </div>
        """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
🤖 AI-Based Employee Attrition Prediction System
<br>
Built with Python • Machine Learning • Streamlit
</div>
""", unsafe_allow_html=True)
