# Student Social Media & Mental Health Impact

# 🎥 Project Demo

Watch the system in action!

A machine learning project that analyzes the relationship between student social media usage and mental health, featuring both predictive models and an interactive web dashboard.

## Project Structure

```
├── data/                   # Dataset
│   └── Student Social Media And Mental Health Impact.csv
├── scripts/                # Analysis & training scripts
│   ├── analysis.py         # EDA: distributions, correlation heatmap
│   └── train_models.py     # CatBoost classifier + regressor training
├── app/                    # Streamlit web application
│   └── app.py              # Interactive prediction dashboard
├── outputs/                # Generated plots & model artifacts
│   ├── target_distributions.png
│   ├── correlation_heatmap.png
│   └── catboost_info/
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Exploratory Data Analysis

```bash
cd scripts
python analysis.py
```

Generates:
- `outputs/target_distributions.png` — histogram of Mental Health Score + bar chart of Stress Level
- `outputs/correlation_heatmap.png` — annotated heatmap of numeric features

### 2. Model Training & Evaluation

```bash
cd scripts
python train_models.py
```

Trains and evaluates two CatBoost models:
- **Classifier** — predicts `Stress_Level` (Low / Medium / High / Very High)
- **Regressor** — predicts `Mental_Health_Score` (0–10)

Prints accuracy, classification report, RMSE, MAE, and R².

### 3. Interactive Dashboard

```bash
cd app
streamlit run app.py
```

A web app where you can:
- Adjust student profile sliders (age, screen time, sleep, study, activity)
- Get real-time predictions for mental health score (gauge chart) and stress level
- Receive personalized actionable insights based on your inputs

## Features

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | 18–24 |
| Gender | Categorical | Male / Female |
| Country | Categorical | 100+ countries |
| Academic Level | Categorical | High School / Undergraduate / Graduate |
| Most Used Platform | Categorical | 12 platforms (Facebook, Instagram, TikTok, etc.) |
| Purpose of Use | Categorical | Networking / Education / Entertainment / News |
| Avg Daily Usage Hours | Numeric | 1.0–8.8 hours |
| Daily Unlocks | Numeric | 62–273 unlocks/day |
| Study Hours | Numeric | 0.3–8.3 hours |
| Physical Activity Hours | Numeric | 0.0–4.1 hours |
| Sleep Hours Per Night | Numeric | 3.6–9.9 hours |
| **Stress Level** (target) | Categorical | Low / Medium / High / Very High |
| **Mental Health Score** (target) | Numeric | 3.6–9.4 (scale of 10) |

## Models

Both models use **CatBoost** with automatic categorical feature handling:
- `CatBoostClassifier` for multi-class stress level prediction
- `CatBoostRegressor` for mental health score regression

## License

This project is for educational/research purposes.
