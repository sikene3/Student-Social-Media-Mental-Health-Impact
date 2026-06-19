import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from catboost import CatBoostClassifier, CatBoostRegressor

st.set_page_config(
    page_title="Mental Health Predictor",
    page_icon="🧠",
    layout="wide",
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Student Social Media And Mental Health Impact.csv"

# ── 1. Cached model training ─────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv(DATA_PATH)

    target_class = "Stress_Level"
    target_reg = "Mental_Health_Score"

    X = df.drop(columns=[target_class, target_reg])
    y_class = df[target_class]
    y_reg = df[target_reg]

    cat_features = [col for col in X.columns if X[col].dtype in ("object", "string")]

    clf = CatBoostClassifier(cat_features=cat_features, verbose=0, random_seed=42)
    clf.fit(X, y_class)

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0, random_seed=42)
    reg.fit(X, y_reg)

    return clf, reg, cat_features, X.columns.tolist(), df

clf, reg, cat_features, feature_cols, df = load_and_train()

# ── 2. Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 Student Profile & Habits")

    st.markdown("### Demographics")
    age = st.slider("Age", min_value=18, max_value=24, value=20)
    gender = st.selectbox("Gender", sorted(df["Gender"].dropna().unique()))
    country = st.selectbox("Country", sorted(df["Country"].dropna().unique()))
    academic = st.selectbox("Academic Level", sorted(df["Academic_Level"].dropna().unique()))

    st.markdown("### Social Media")
    platform = st.selectbox("Most Used Platform", sorted(df["Most_Used_Platform"].dropna().unique()))
    purpose = st.selectbox("Purpose of Use", sorted(df["Purpose_Of_Use"].dropna().unique()))
    usage_hours = st.slider("Avg Daily Usage (hours)", min_value=1.0, max_value=9.0, value=5.0, step=0.1)
    unlocks = st.slider("Daily Unlocks", min_value=60, max_value=275, value=170)

    st.markdown("### Lifestyle")
    study = st.slider("Study Hours", min_value=0.0, max_value=8.5, value=3.0, step=0.1)
    activity = st.slider("Physical Activity (hours)", min_value=0.0, max_value=4.5, value=1.5, step=0.1)
    sleep = st.slider("Sleep Per Night (hours)", min_value=3.5, max_value=10.0, value=6.5, step=0.1)

# ── 3. Main area ─────────────────────────────────────────────────────────────
st.title("🧠 Student Mental Health & Stress Predictor")
st.markdown(
    "Adjust your profile in the sidebar and click **Analyze** to get "
    "AI‑powered predictions of your mental health score and stress level."
)

if st.button("🔍 Analyze Mental Health", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Country": country,
        "Academic_Level": academic,
        "Most_Used_Platform": platform,
        "Purpose_Of_Use": purpose,
        "Avg_Daily_Usage_Hours": usage_hours,
        "Daily_Unlocks": unlocks,
        "Study_Hours": study,
        "Physical_Activity_Hours": activity,
        "Sleep_Hours_Per_Night": sleep,
    }])

    input_data = input_data[feature_cols]

    mh_score = float(reg.predict(input_data).ravel()[0])
    stress_level = str(clf.predict(input_data).ravel()[0])

    # ── Gauge + Stress columns ────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=mh_score,
                number={"font": {"size": 48}, "suffix": " / 10"},
                delta={"reference": df["Mental_Health_Score"].mean(),
                       "increasing": {"color": "green"},
                       "decreasing": {"color": "red"}},
                gauge={
                    "axis": {"range": [0, 10], "tickwidth": 1},
                    "bar": {"color": "royalblue"},
                    "steps": [
                        {"range": [0, 3], "color": "tomato"},
                        {"range": [3, 5], "color": "orange"},
                        {"range": [5, 7], "color": "gold"},
                        {"range": [7, 10], "color": "mediumseagreen"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": mh_score,
                    },
                },
                title={"text": "Mental Health Score"},
            )
        )
        fig.update_layout(height=350, margin=dict(t=60, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        stress_color = {
            "Low": "success",
            "Medium": "warning",
            "High": "error",
            "Very High": "error",
        }
        level = stress_color.get(stress_level, "info")

        if level == "success":
            st.success(f"### Stress Level: {stress_level}")
        elif level == "warning":
            st.warning(f"### Stress Level: {stress_level}")
        elif level == "error":
            st.error(f"### Stress Level: {stress_level}")
        else:
            st.info(f"### Stress Level: {stress_level}")

        st.markdown("---")
        st.markdown("#### 📊 Your Inputs at a Glance")
        st.caption(f"🕐 {usage_hours:.1f}h screen time  •  📱 {unlocks} unlocks/day")
        st.caption(f"📚 {study:.1f}h study  •  🏃 {activity:.1f}h activity  •  😴 {sleep:.1f}h sleep")

    # ── 4. Actionable Insights ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💡 Actionable Insights")

    tips = []

    if stress_level in ("High", "Very High"):
        tips.append(
            "**Consider a digital detox:** Your stress level is elevated. "
            "Try reducing daily screen time by 1–2 hours and disabling non‑essential notifications."
        )
        if sleep < 7:
            tips.append(
                "**Prioritize sleep:** You're averaging less than 7 hours per night. "
                "Aim for 7–9 hours — consistent sleep is one of the strongest buffers against stress."
            )
        if activity < 1.5:
            tips.append(
                "**Move more:** Even 30 minutes of light physical activity (walking, stretching) "
                "can significantly lower cortisol levels and improve mood."
            )
        if usage_hours > 6:
            tips.append(
                "**Set app limits:** You're spending over 6 hours daily on social media. "
                "Use built‑in screen‑time tools to cap usage at 4–5 hours."
            )

    elif stress_level == "Medium":
        tips.append(
            "**Stay mindful:** Your stress is moderate. Practice short mindfulness breaks "
            "(5 minutes of deep breathing) between study sessions to keep it in check."
        )
        if sleep < 7:
            tips.append(
                "**Guard your sleep:** Try winding down 30 minutes earlier — "
                "even small improvements in sleep quality can reduce medium‑grade stress."
            )
        if activity < 2:
            tips.append(
                "**Stay active:** Regular exercise helps prevent stress from escalating. "
                "Aim for at least 2 hours of physical activity per day."
            )

    else:  # Low
        tips.append(
            "**Keep it up!** Your current habits are supporting a low stress level. "
            "Maintain your balance of study, activity, and sleep to stay in this healthy zone."
        )
        tips.append(
            "**Share your routine:** Your approach is working — consider mentoring a friend "
            "who might be struggling with time management or screen habits."
        )

    for i, tip in enumerate(tips, 1):
        if stress_level in ("High", "Very High"):
            st.warning(f"{tip}")
        elif stress_level == "Medium":
            st.info(f"{tip}")
        else:
            st.success(f"{tip}")
