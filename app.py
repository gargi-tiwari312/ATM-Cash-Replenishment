import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date as dt_date


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ATM Cash Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_FILE = "ATM_Cash_Replenishment_Model.pkl"

try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    st.error(
        f"Model file '{MODEL_FILE}' was not found. "
        "Keep the .pkl file in the same folder as app.py."
    )
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #f4f7fb;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: #edf5ff;
    border-right: 1px solid #d8e6f5;
}

.hero {
    background: linear-gradient(
        135deg,
        #082b63 0%,
        #0d47a1 48%,
        #2196f3 100%
    );
    color: white;
    padding: 30px 34px;
    border-radius: 22px;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(13, 71, 161, 0.20);
}

.hero h1 {
    margin: 0;
    font-size: 34px;
    font-weight: 750;
}

.hero p {
    margin: 8px 0 0 0;
    font-size: 16px;
    opacity: 0.92;
}

.badge {
    display: inline-block;
    margin-top: 15px;
    padding: 6px 12px;
    border-radius: 20px;
    background: rgba(255,255,255,0.15);
    font-size: 13px;
}

.section-title {
    color: #082b63;
    font-size: 22px;
    font-weight: 750;
    margin-top: 24px;
    margin-bottom: 12px;
}

.card {
    background: white;
    border: 1px solid #e3eaf2;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(24, 55, 90, 0.07);
}

.status-card {
    background: white;
    border: 1px solid #e3eaf2;
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 5px 18px rgba(24, 55, 90, 0.07);
}

.recommendation {
    background: linear-gradient(135deg, #ffffff, #f4f9ff);
    border-left: 6px solid #1976d2;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 5px 18px rgba(24, 55, 90, 0.08);
}

.footer {
    text-align: center;
    color: #718096;
    padding: 25px 0 5px 0;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e3eaf2;
    padding: 14px;
    border-radius: 15px;
    box-shadow: 0 5px 18px rgba(24, 55, 90, 0.06);
}

.small-text {
    color: #718096;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MAPS - KEEP CONSISTENT WITH TRAINING
# =========================================================

ATM_MAP = {
    f"ATM_{i:04d}": i - 1
    for i in range(1, 51)
}

DAY_MAP = {
    "Monday": 1,
    "Tuesday": 5,
    "Wednesday": 6,
    "Thursday": 4,
    "Friday": 0,
    "Saturday": 2,
    "Sunday": 3
}

TIME_MAP = {
    "Morning": 2,
    "Afternoon": 0,
    "Evening": 1,
    "Night": 3
}

LOCATION_MAP = {
    "Bank Branch": 0,
    "Gas Station": 1,
    "Mall": 2,
    "Standalone": 3,
    "Supermarket": 4
}

WEATHER_MAP = {
    "Clear": 0,
    "Cloudy": 1,
    "Rainy": 2,
    "Snowy": 3
}

ATM_LIST = list(ATM_MAP.keys())

DAYS = list(DAY_MAP.keys())

TIMES = list(TIME_MAP.keys())

LOCATIONS = list(LOCATION_MAP.keys())

WEATHER = list(WEATHER_MAP.keys())


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def create_input(
    selected_atm,
    selected_day,
    selected_time,
    selected_location,
    selected_weather,
    selected_holiday,
    selected_event,
    selected_withdrawals,
    selected_deposits,
    selected_previous_cash,
    selected_nearby,
    selected_year,
    selected_month,
    selected_day_number
):
    return pd.DataFrame([{
        "ATM_ID": ATM_MAP[selected_atm],
        "Day_of_Week": DAY_MAP[selected_day],
        "Time_of_Day": TIME_MAP[selected_time],
        "Total_Withdrawals": selected_withdrawals,
        "Total_Deposits": selected_deposits,
        "Location_Type": LOCATION_MAP[selected_location],
        "Holiday_Flag": selected_holiday,
        "Special_Event_Flag": selected_event,
        "Previous_Day_Cash_Level": selected_previous_cash,
        "Weather_Condition": WEATHER_MAP[selected_weather],
        "Nearby_Competitor_ATMs": selected_nearby,
        "Year": selected_year,
        "Month": selected_month,
        "Day": selected_day_number
    }])


def classify_demand(value):
    if value < 30000:
        return (
            "🟢 LOW",
            "SAFE",
            "Low demand expected. The current refill schedule "
            "should be sufficient."
        )

    if value < 70000:
        return (
            "🟡 MEDIUM",
            "MONITOR",
            "Moderate demand expected. Maintain the standard "
            "replenishment schedule."
        )

    return (
        "🔴 HIGH",
        "REFILL REQUIRED",
        "High demand expected. ATM replenishment should be prioritized."
    )


def currency(value):
    return f"₹{value:,.0f}"


def priority_score(prediction, previous_cash):
    """
    A simple operational priority score for demonstration.
    It is a decision-support score, not an ML output.
    """
    if prediction <= 0:
        return 0

    demand_component = min(prediction / 100000, 1) * 50

    gap = max(prediction - previous_cash, 0)

    gap_component = min(gap / max(prediction, 1), 1) * 50

    return int(round(demand_component + gap_component))


def make_prediction(
    selected_atm,
    selected_day,
    selected_time,
    selected_location,
    selected_weather,
    selected_holiday,
    selected_event,
    selected_withdrawals,
    selected_deposits,
    selected_previous_cash,
    selected_nearby,
    selected_year,
    selected_month,
    selected_day_number
):
    X = create_input(
        selected_atm,
        selected_day,
        selected_time,
        selected_location,
        selected_weather,
        selected_holiday,
        selected_event,
        selected_withdrawals,
        selected_deposits,
        selected_previous_cash,
        selected_nearby,
        selected_year,
        selected_month,
        selected_day_number
    )

    return float(model.predict(X)[0])


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<h1>🏦 ATM Cash Intelligence Dashboard</h1>

<p>
Machine Learning Based Cash Demand Forecasting & Refill Optimization
</p>

<div class="badge">
MCA (Data Science) • Predictive Analytics • Streamlit
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🏦 ATM Control Center")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔮 Prediction",
        "🏧 ATM Monitoring",
        "📈 Analytics",
        "📋 Reports"
    ]
)

st.sidebar.markdown("---")


# =========================================================
# COMMON INPUTS
# =========================================================

st.sidebar.subheader("🏧 ATM Information")

atm = st.sidebar.selectbox(
    "ATM ID",
    ATM_LIST,
    index=40
)

location = st.sidebar.selectbox(
    "Location Type",
    LOCATIONS
)

nearby = st.sidebar.number_input(
    "Nearby Competitor ATMs",
    min_value=0,
    value=5
)

st.sidebar.markdown("---")

st.sidebar.subheader("💳 Transaction Details")

withdrawals = st.sidebar.number_input(
    "Total Withdrawals",
    min_value=0,
    value=57450
)

deposits = st.sidebar.number_input(
    "Total Deposits",
    min_value=0,
    value=9308
)

previous_cash = st.sidebar.number_input(
    "Previous Day Cash Level",
    min_value=0,
    value=112953
)

st.sidebar.markdown("---")

st.sidebar.subheader("🌦 Context")

day = st.sidebar.selectbox(
    "Day of Week",
    DAYS
)

time = st.sidebar.selectbox(
    "Time of Day",
    TIMES
)

weather = st.sidebar.selectbox(
    "Weather",
    WEATHER
)

holiday = st.sidebar.selectbox(
    "Holiday",
    [0, 1]
)

event = st.sidebar.selectbox(
    "Special Event",
    [0, 1]
)

st.sidebar.markdown("---")

st.sidebar.subheader("📅 Prediction Date")

selected_date = st.sidebar.date_input(
    "Date",
    value=dt_date(2022, 4, 25)
)

year = selected_date.year
month = selected_date.month
date_number = selected_date.day

st.sidebar.markdown("---")

st.sidebar.subheader("🛡 Refill Policy")

safety_buffer = st.sidebar.slider(
    "Safety Buffer",
    min_value=5,
    max_value=30,
    value=20,
    step=5,
    format="%d%%"
)

st.sidebar.caption(
    "The safety buffer is added to predicted demand "
    "to reduce shortage risk."
)


# =========================================================
# PREDICT BUTTON
# =========================================================

predict_button = st.sidebar.button(
    "🔮 RUN PREDICTION",
    use_container_width=True,
    type="primary"
)


# =========================================================
# RUN PREDICTION
# =========================================================

if predict_button:

    with st.spinner("Running Machine Learning prediction..."):

        prediction = make_prediction(
            atm,
            day,
            time,
            location,
            weather,
            holiday,
            event,
            withdrawals,
            deposits,
            previous_cash,
            nearby,
            year,
            month,
            date_number
        )

    refill = prediction * (1 + safety_buffer / 100)

    demand, risk, message = classify_demand(prediction)

    cash_gap = prediction - previous_cash

    priority = priority_score(
        prediction,
        previous_cash
    )

    result = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "atm": atm,
        "prediction": prediction,
        "refill": refill,
        "demand": demand,
        "risk": risk,
        "cash_gap": cash_gap,
        "priority": priority
    }

    st.session_state.last_result = result

    st.session_state.history.append(result)


# =========================================================
# DASHBOARD PAGE
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">📊 ATM Network Overview</div>',
        unsafe_allow_html=True
    )

    total_atms = 50

    if st.session_state.last_result:
        result = st.session_state.last_result

        current_prediction = result["prediction"]

        high_risk = (
            1 if result["risk"] == "REFILL REQUIRED" else 0
        )

        total_refill = result["refill"]

    else:
        current_prediction = 0
        high_risk = 0
        total_refill = 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🏧 Total ATMs",
        total_atms
    )

    c2.metric(
        "🔴 Current High Risk",
        high_risk
    )

    c3.metric(
        "💰 Current Predicted Demand",
        currency(current_prediction)
    )

    c4.metric(
        "🚚 Current Recommended Refill",
        currency(total_refill)
    )

    st.markdown(
        '<div class="section-title">⚡ Quick Actions</div>',
        unsafe_allow_html=True
    )

    q1, q2, q3 = st.columns(3)

    with q1:
        st.markdown("""
        <div class="card">
        <h3>🔮 Prediction</h3>
        <p>Forecast the next-day cash requirement for a selected ATM.</p>
        </div>
        """, unsafe_allow_html=True)

    with q2:
        st.markdown("""
        <div class="card">
        <h3>🏧 Monitoring</h3>
        <p>Compare ATM demand scenarios and identify high-priority ATMs.</p>
        </div>
        """, unsafe_allow_html=True)

    with q3:
        st.markdown("""
        <div class="card">
        <h3>📋 Reports</h3>
        <p>Review prediction history and download operational reports.</p>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.last_result:

        result = st.session_state.last_result

        st.markdown(
            '<div class="section-title">🚦 Latest ATM Status</div>',
            unsafe_allow_html=True
        )

        s1, s2, s3 = st.columns(3)

        with s1:
            st.markdown(
                f"""
                <div class="status-card">
                    <h2>{result["risk"]}</h2>
                    <p>ATM Risk Status</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with s2:
            st.markdown(
                f"""
                <div class="status-card">
                    <h2>{result["priority"]}/100</h2>
                    <p>Refill Priority Score</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with s3:
            gap_text = (
                "Additional cash required"
                if result["cash_gap"] > 0
                else "Current cash is sufficient"
            )

            st.markdown(
                f"""
                <div class="status-card">
                    <h2>{currency(result["cash_gap"])}</h2>
                    <p>{gap_text}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.info(
            "No prediction has been generated yet. "
            "Enter the inputs in the sidebar and click "
            "'RUN PREDICTION'."
        )


# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "🔮 Prediction":

    st.markdown(
        '<div class="section-title">🔮 ATM Cash Demand Prediction</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.last_result:

        st.info(
            "Click 'RUN PREDICTION' from the sidebar to generate "
            "the prediction dashboard."
        )

    else:

        result = st.session_state.last_result

        prediction = result["prediction"]
        refill = result["refill"]
        demand = result["demand"]
        risk = result["risk"]
        cash_gap = result["cash_gap"]
        priority = result["priority"]

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "💰 Predicted Demand",
            currency(prediction)
        )

        k2.metric(
            "💵 Recommended Refill",
            currency(refill)
        )

        k3.metric(
            "📈 Demand Level",
            demand
        )

        k4.metric(
            "🎯 Priority Score",
            f"{priority}/100"
        )

        st.markdown(
            '<div class="section-title">💰 Cash Gap Analysis</div>',
            unsafe_allow_html=True
        )

        g1, g2 = st.columns(2)

        with g1:
            st.metric(
                "Predicted Demand - Previous Cash",
                currency(cash_gap)
            )

        with g2:

            if cash_gap > 0:
                st.error(
                    "🔴 Additional cash is required."
                )
            else:
                st.success(
                    "🟢 Previous cash is sufficient."
                )

        st.markdown(
            '<div class="section-title">📊 Cash Comparison</div>',
            unsafe_allow_html=True
        )

        chart_data = pd.DataFrame({
            "Category": [
                "Previous Cash",
                "Predicted Demand",
                "Recommended Refill"
            ],
            "Amount": [
                previous_cash,
                prediction,
                refill
            ]
        })

        fig = px.bar(
            chart_data,
            x="Category",
            y="Amount",
            color="Category",
            text="Amount",
            title="Cash Requirement Comparison"
        )

        fig.update_traces(
            texttemplate="₹%{y:,.0f}",
            textposition="outside"
        )

        fig.update_layout(
            showlegend=False,
            yaxis_title="Cash Amount (₹)",
            xaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown(
            '<div class="section-title">🎯 Demand Gauge</div>',
            unsafe_allow_html=True
        )

        gauge_max = max(
            100000,
            prediction * 1.2
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={
                    "text": "Predicted Cash Demand"
                },
                number={
                    "prefix": "₹"
                },
                gauge={
                    "axis": {
                        "range": [
                            0,
                            gauge_max
                        ]
                    },
                    "bar": {
                        "color": "#1976D2"
                    },
                    "steps": [
                        {
                            "range": [
                                0,
                                min(30000, gauge_max)
                            ],
                            "color": "#E8F5E9"
                        },
                        {
                            "range": [
                                min(30000, gauge_max),
                                min(70000, gauge_max)
                            ],
                            "color": "#FFF8E1"
                        },
                        {
                            "range": [
                                min(70000, gauge_max),
                                gauge_max
                            ],
                            "color": "#FFEBEE"
                        }
                    ]
                }
            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        st.markdown(
            '<div class="section-title">💡 Smart Recommendation</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="recommendation">

            <h3>ATM Replenishment Recommendation</h3>

            <p><b>ATM:</b> {atm}</p>

            <p><b>Predicted Demand:</b>
            {currency(prediction)}</p>

            <p><b>Safety Buffer:</b>
            {safety_buffer}%</p>

            <p><b>Recommended Refill:</b>
            {currency(refill)}</p>

            <p><b>Demand:</b> {demand}</p>

            <p><b>Risk Status:</b> {risk}</p>

            <p><b>Recommendation:</b> {message}</p>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# ATM MONITORING PAGE
# =========================================================

elif page == "🏧 ATM Monitoring":

    st.markdown(
        '<div class="section-title">🏧 ATM Network Monitoring</div>',
        unsafe_allow_html=True
    )

    st.info(
        "This page performs a scenario-based comparison of the 50 ATMs "
        "using the current transaction/context inputs while changing "
        "only the ATM ID. For a production system, each ATM would use "
        "its own live transaction and cash-level data."
    )

    if st.button(
        "🔄 Analyze All 50 ATMs",
        type="primary"
    ):

        rows = []

        progress = st.progress(0)

        for index, selected_atm in enumerate(ATM_LIST):

            atm_input = create_input(
                selected_atm,
                day,
                time,
                location,
                weather,
                holiday,
                event,
                withdrawals,
                deposits,
                previous_cash,
                nearby,
                year,
                month,
                date_number
            )

            atm_prediction = float(
                model.predict(atm_input)[0]
            )

            atm_refill = atm_prediction * (
                1 + safety_buffer / 100
            )

            atm_gap = (
                atm_prediction - previous_cash
            )

            atm_priority = priority_score(
                atm_prediction,
                previous_cash
            )

            if atm_prediction < 30000:
                atm_status = "🟢 Safe"
            elif atm_prediction < 70000:
                atm_status = "🟡 Monitor"
            else:
                atm_status = "🔴 Refill Required"

            rows.append({
                "ATM ID": selected_atm,
                "Predicted Demand": atm_prediction,
                "Recommended Refill": atm_refill,
                "Cash Gap": atm_gap,
                "Priority Score": atm_priority,
                "Status": atm_status
            })

            progress.progress(
                (index + 1) / len(ATM_LIST)
            )

        monitoring_df = pd.DataFrame(rows)

        monitoring_df = monitoring_df.sort_values(
            "Priority Score",
            ascending=False
        )

        st.session_state["monitoring_df"] = monitoring_df

    if "monitoring_df" in st.session_state:

        monitoring_df = st.session_state["monitoring_df"]

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "🏧 ATMs Analyzed",
            len(monitoring_df)
        )

        m2.metric(
            "🔴 High Priority",
            int(
                (monitoring_df["Priority Score"] >= 70).sum()
            )
        )

        m3.metric(
            "🟡 Medium Priority",
            int(
                (
                    (monitoring_df["Priority Score"] >= 40)
                    &
                    (monitoring_df["Priority Score"] < 70)
                ).sum()
            )
        )

        m4.metric(
            "🟢 Low Priority",
            int(
                (monitoring_df["Priority Score"] < 40).sum()
            )
        )

        st.markdown(
            '<div class="section-title">🚚 Refill Priority Ranking</div>',
            unsafe_allow_html=True
        )

        top5 = monitoring_df.head(5)

        top5_display = top5.copy()

        top5_display["Predicted Demand"] = (
            top5_display["Predicted Demand"]
            .map(currency)
        )

        top5_display["Recommended Refill"] = (
            top5_display["Recommended Refill"]
            .map(currency)
        )

        top5_display["Cash Gap"] = (
            top5_display["Cash Gap"]
            .map(currency)
        )

        st.dataframe(
            top5_display,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            '<div class="section-title">📊 Priority Distribution</div>',
            unsafe_allow_html=True
        )

        priority_chart = px.bar(
            monitoring_df.head(15),
            x="ATM ID",
            y="Priority Score",
            color="Priority Score",
            title="Top ATM Refill Priority Scores"
        )

        priority_chart.update_layout(
            xaxis_title="ATM",
            yaxis_title="Priority Score"
        )

        st.plotly_chart(
            priority_chart,
            use_container_width=True
        )

        st.markdown(
            '<div class="section-title">📋 All ATM Results</div>',
            unsafe_allow_html=True
        )

        display_df = monitoring_df.copy()

        display_df["Predicted Demand"] = (
            display_df["Predicted Demand"]
            .map(currency)
        )

        display_df["Recommended Refill"] = (
            display_df["Recommended Refill"]
            .map(currency)
        )

        display_df["Cash Gap"] = (
            display_df["Cash Gap"]
            .map(currency)
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        monitoring_csv = monitoring_df.to_csv(
            index=False
        )

        st.download_button(
            "⬇ Download ATM Monitoring Report",
            monitoring_csv,
            "ATM_Network_Monitoring.csv",
            "text/csv"
        )


# =========================================================
# ANALYTICS PAGE
# =========================================================

elif page == "📈 Analytics":

    st.markdown(
        '<div class="section-title">📈 Machine Learning Analytics</div>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs([
        "🔍 Feature Influence",
        "📅 Scenario Forecast",
        "🤖 Model Performance"
    ])


    # FEATURE INFLUENCE

    with tab1:

        st.subheader(
            "🔍 Linear Regression Feature Influence"
        )

        if hasattr(model, "coef_"):

            feature_names = [
                "ATM_ID",
                "Day_of_Week",
                "Time_of_Day",
                "Total_Withdrawals",
                "Total_Deposits",
                "Location_Type",
                "Holiday_Flag",
                "Special_Event_Flag",
                "Previous_Day_Cash_Level",
                "Weather_Condition",
                "Nearby_Competitor_ATMs",
                "Year",
                "Month",
                "Day"
            ]

            coefficients = np.asarray(
                model.coef_
            ).reshape(-1)

            feature_df = pd.DataFrame({
                "Feature": feature_names,
                "Coefficient": coefficients
            })

            feature_df["Absolute Influence"] = (
                feature_df["Coefficient"].abs()
            )

            feature_df = feature_df.sort_values(
                "Absolute Influence",
                ascending=False
            )

            feature_fig = px.bar(
                feature_df,
                x="Absolute Influence",
                y="Feature",
                orientation="h",
                title="Relative Feature Influence"
            )

            st.plotly_chart(
                feature_fig,
                use_container_width=True
            )

            st.caption(
                "Coefficient magnitude indicates relative model influence "
                "for the trained Linear Regression model. It does not prove causation."
            )

        else:

            st.info(
                "The loaded model does not expose Linear Regression coefficients."
            )


    # SCENARIO FORECAST

    with tab2:

        st.subheader(
            "📅 7-Day Scenario Forecast"
        )

        st.info(
            "This is a scenario analysis. It changes the day-of-week "
            "input while keeping the current transaction/context values fixed."
        )

        forecast_rows = []

        for forecast_day in DAYS:

            forecast_input = create_input(
                atm,
                forecast_day,
                time,
                location,
                weather,
                holiday,
                event,
                withdrawals,
                deposits,
                previous_cash,
                nearby,
                year,
                month,
                date_number
            )

            forecast_prediction = float(
                model.predict(forecast_input)[0]
            )

            forecast_rows.append({
                "Day": forecast_day,
                "Predicted Demand": forecast_prediction
            })

        forecast_df = pd.DataFrame(
            forecast_rows
        )

        forecast_fig = px.line(
            forecast_df,
            x="Day",
            y="Predicted Demand",
            markers=True,
            title="Scenario-Based Weekly Cash Demand"
        )

        forecast_fig.update_layout(
            yaxis_title="Cash Demand (₹)",
            xaxis_title=""
        )

        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )

        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True
        )


    # MODEL PERFORMANCE

    with tab3:

        st.subheader(
            "🤖 Model Performance"
        )

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "Training R²",
            "0.863"
        )

        p2.metric(
            "Testing R²",
            "0.871"
        )

        p3.metric(
            "Algorithm",
            "Linear Regression"
        )

        st.markdown(
            """
            <div class="card">

            <h3>Model Interpretation</h3>

            <p>
            The trained Linear Regression model predicts the continuous
            target variable <b>Cash_Demand_Next_Day</b>.
            </p>

            <p>
            The model was selected because it is simple, fast,
            interpretable, and suitable for a regression problem.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# REPORTS PAGE
# =========================================================

elif page == "📋 Reports":

    st.markdown(
        '<div class="section-title">📋 Reports & Prediction History</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.history:

        st.info(
            "No prediction history is available yet. "
            "Run at least one prediction."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        display_history = history_df.copy()

        display_history["prediction"] = (
            display_history["prediction"]
            .map(currency)
        )

        display_history["refill"] = (
            display_history["refill"]
            .map(currency)
        )

        display_history["cash_gap"] = (
            display_history["cash_gap"]
            .map(currency)
        )

        display_history = display_history.rename(
            columns={
                "time": "Time",
                "atm": "ATM ID",
                "prediction": "Predicted Demand",
                "refill": "Recommended Refill",
                "demand": "Demand Level",
                "risk": "Risk Status",
                "cash_gap": "Cash Gap",
                "priority": "Priority Score"
            }
        )

        st.dataframe(
            display_history,
            use_container_width=True,
            hide_index=True
        )

        report_csv = history_df.to_csv(
            index=False
        )

        st.download_button(
            "⬇ Download Prediction History",
            report_csv,
            "ATM_Prediction_History.csv",
            "text/csv"
        )


# =========================================================
# MODEL INFORMATION
# =========================================================

with st.expander("🤖 About the Project & Model"):

    info1, info2 = st.columns(2)

    with info1:

        st.markdown(
            """
            <div class="card">

            <h3>Project</h3>

            <p>
            <b>ATM Cash Replenishment Prediction System</b>
            </p>

            <p>
            The system uses historical ATM-related features to
            estimate next-day cash demand and support refill decisions.
            </p>

            <p>
            <b>Application:</b> Streamlit
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with info2:

        st.markdown(
            """
            <div class="card">

            <h3>Machine Learning</h3>

            <p><b>Algorithm:</b> Linear Regression</p>
            <p><b>Target:</b> Cash_Demand_Next_Day</p>
            <p><b>Training R²:</b> 0.863</p>
            <p><b>Testing R²:</b> 0.871</p>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div class="footer">

<h4>🏦 ATM Cash Intelligence Dashboard</h4>

<p>
Python • Pandas • NumPy • Scikit-learn • Joblib • Streamlit • Plotly
</p>

<p>
MCA (Data Science) Final Year Project | Gargi Tiwari
</p>

</div>
""", unsafe_allow_html=True)

