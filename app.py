import pandas as pd
import streamlit as st
import traceback
import matplotlib.pyplot as plt

try:
    import joblib
except ImportError:
    joblib = None


def load_serialized_object(path):
    if joblib is None:
        return None, "joblib not installed"
    try:
        return joblib.load(path), None
    except BaseException:
        return None, traceback.format_exc()


class DummyScaler:
    def transform(self, X):
        return X.copy()


class DummyModel:
    def predict(self, X):
        if hasattr(X, 'iloc'):
            X = X.reset_index(drop=True)
        base = 20
        score = (
            base
            + X['temp'].astype(float) * 30
            + X['atemp'].astype(float) * 15
            - X['hum'].astype(float) * 5
            - X['windspeed'].astype(float) * 10
            + X['rush_hour_Rush Hour'].astype(int) * 25
        )
        return score.clip(lower=0).round().astype(int).values


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Bike Rental Demand Predictor",
    page_icon="🚲",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

model, model_error = load_serialized_object("bike_rental_model.pkl")
scaler, scaler_error = load_serialized_object("scaler.pkl")

fallback_mode = False
model_name = "XGBoost Regressor"

if model_error or scaler_error:
    fallback_mode = True
    model_name = "Fallback predictor"
    st.warning(
        "A fallback prediction engine is active because the saved model or scaler could not be loaded. "
        "Predictions are approximate and for UI demonstration only."
    )
    model = model or DummyModel()
    scaler = scaler or DummyScaler()

# =========================
# SIDEBAR INPUTS
# =========================

st.sidebar.header("Scenario Inputs")

year = st.sidebar.selectbox(
    "Year",
    [2011, 2012]
)

month = st.sidebar.slider(
    "Month",
    1,
    12,
    6
)

hr = st.sidebar.slider(
    "Hour",
    0,
    23,
    8
)

holiday = st.sidebar.selectbox(
    "Holiday",
    ["No", "Yes"]
)

weekday = st.sidebar.slider(
    "Weekday (0=Sunday)",
    0,
    6,
    1
)

workingday = st.sidebar.selectbox(
    "Working Day",
    ["No", "Yes"]
)

temp = st.sidebar.slider(
    "Temperature (norm.)",
    0.0,
    1.0,
    0.60
)

atemp = st.sidebar.slider(
    "Feels Like Temp (norm.)",
    0.0,
    1.0,
    0.62
)

hum = st.sidebar.slider(
    "Humidity (norm.)",
    0.0,
    1.0,
    0.50
)

windspeed = st.sidebar.slider(
    "Wind Speed (norm.)",
    0.0,
    1.0,
    0.20
)

season = st.sidebar.selectbox(
    "Season",
    ["spring", "summer", "fall", "winter"]
)

weather = st.sidebar.selectbox(
    "Weather",
    ["Clear", "Mist", "Light Snow", "Heavy Rain"]
)

rush_hour = st.sidebar.selectbox(
    "Rush Hour",
    ["No", "Yes"]
)

predict_button = st.sidebar.button(
    "Predict Demand"
)

# =========================
# HEADER
# =========================

st.title("🚲 Bike Rental Demand Predictor")

st.caption(
    "Predict hourly bike demand using weather, season and calendar features."
)

# =========================
# MODEL COMPARISON DATA
# =========================

comparison_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost"
    ],
    "MAE": [
        86.164766,
        30.324263,
        24.490868,
        25.672372,
        24.696899
    ],
    "RMSE": [
        114.498480,
        50.843465,
        41.209217,
        41.050490,
        39.702153
    ],
    "R2 Score": [
        0.585987,
        0.918363,
        0.946371,
        0.946783,
        0.950221
    ]
})

# =========================
# DEFAULT VALUE
# =========================

predicted_value = None

# =========================
# PREDICTION
# =========================

if predict_button:

    if fallback_mode:
        st.warning(
            "The saved model or scaler could not be loaded. Using fallback prediction logic. "
            "Results are approximate and for demonstration only."
        )

    input_data = {
        'hr': hr,
        'weekday': weekday,
        'temp': temp,
        'atemp': atemp,
        'hum': hum,
        'windspeed': windspeed,

        'season_springer': 1 if season == "spring" else 0,
        'season_summer': 1 if season == "summer" else 0,
        'season_winter': 1 if season == "winter" else 0,

        'yr_2012': 1 if year == 2012 else 0,

        'mnth_10': 1 if month == 10 else 0,
        'mnth_11': 1 if month == 11 else 0,
        'mnth_12': 1 if month == 12 else 0,
        'mnth_2': 1 if month == 2 else 0,
        'mnth_3': 1 if month == 3 else 0,
        'mnth_4': 1 if month == 4 else 0,
        'mnth_5': 1 if month == 5 else 0,
        'mnth_6': 1 if month == 6 else 0,
        'mnth_7': 1 if month == 7 else 0,
        'mnth_8': 1 if month == 8 else 0,
        'mnth_9': 1 if month == 9 else 0,

        'holiday_Yes': 1 if holiday == "Yes" else 0,

        'workingday_Working Day': 1 if workingday == "Yes" else 0,

        'weathersit_Heavy Rain': 1 if weather == "Heavy Rain" else 0,
        'weathersit_Light Snow': 1 if weather == "Light Snow" else 0,
        'weathersit_Mist': 1 if weather == "Mist" else 0,

        'rush_hour_Rush Hour': 1 if rush_hour == "Yes" else 0
    }

    input_df = pd.DataFrame([input_data])

    scaled_cols = [
        'hr',
        'weekday',
        'temp',
        'atemp',
        'hum',
        'windspeed'
    ]

    try:
        input_df[scaled_cols] = scaler.transform(
            input_df[scaled_cols]
        )
        prediction = model.predict(input_df)
        predicted_value = int(prediction[0])
    except Exception:
        predicted_value = None
        st.error(
            "Prediction failed. The app is using fallback prediction logic because the saved artifacts are invalid. "
            "Please restore the saved model and scaler files for real predictions."
        )

# =========================
# TOP SECTION
# =========================

left, right = st.columns([1.2, 1])

with left:

    st.subheader("Prediction")

    if predicted_value is not None:

        st.metric(
            "Predicted Demand",
            f"{predicted_value} bikes/hour"
        )

        st.success(
            f"Model Used : {model_name}"
        )

        if fallback_mode:
            st.info(
                "Note: this result is generated by fallback logic because the saved model/scaler failed to load."
            )

    else:

        st.info(
            "Fill inputs from sidebar and click Predict Demand."
        )

with right:

    st.subheader("Deployed Model")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Best Model",
            "XGBoost"
        )

    with col2:
        st.metric(
            "MAE",
            "24.70"
        )

    with col3:
        st.metric(
            "RMSE",
            "39.70"
        )

    with col4:
        st.metric(
            "R²",
            "0.9502"
        )

st.divider()

# =========================
# MODEL COMPARISON TABLE
# =========================

st.subheader("📊 Model Comparison")

st.table(
    comparison_df.sort_values(
        by="R2 Score",
        ascending=False
    )
)

# =========================
# SELECTED SCENARIO
# =========================

st.subheader("Selected Scenario")

scenario_df = pd.DataFrame({
    "Feature": [
        "Year",
        "Month",
        "Hour",
        "Weekday",
        "Season",
        "Weather",
        "Holiday",
        "Working Day"
    ],
    "Value": [
        year,
        month,
        hr,
        weekday,
        season,
        weather,
        holiday,
        workingday
    ]
})

# Ensure mixed-type columns display without Arrow conversion errors
scenario_df["Value"] = scenario_df["Value"].astype(str)

st.table(
    scenario_df
)
