# ==========================================
# SEO RANKING PREDICTOR - FULL PROJECT
# Production-Ready (ML + API + Dashboard)
# ==========================================

# ==============================
# requirements.txt
# ==============================
# pandas
# numpy
# scikit-learn
# xgboost
# fastapi
# uvicorn
# joblib
# streamlit
# shap

# ==============================
# src/config.py
# ==============================
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "seo_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

# ==============================
# src/data_loader.py
# ==============================
import pandas as pd
import numpy as np
from src.config import DATA_PATH

def generate_data(n=3000):
    df = pd.DataFrame({
        "backlinks": np.random.randint(0, 5000, n),
        "ref_domains": np.random.randint(0, 500, n),
        "content_length": np.random.randint(300, 5000, n),
        "page_speed": np.random.uniform(0.5, 5, n),
        "domain_authority": np.random.randint(1, 100, n),
        "internal_links": np.random.randint(0, 100, n),
        "external_links": np.random.randint(0, 50, n),
    })

    df["ranking"] = (
        100
        - df["backlinks"] * 0.01
        - df["ref_domains"] * 0.05
        - df["domain_authority"] * 0.3
        - df["content_length"] * 0.002
        + df["page_speed"] * 2
        + np.random.normal(0, 5, n)
    ).clip(1, 100)

    df.to_csv(DATA_PATH, index=False)
    return df

# ==============================
# src/feature_engineering.py
# ==============================
def engineer(df):
    df["link_ratio"] = df["external_links"] / (df["internal_links"] + 1)
    df["authority_score"] = df["domain_authority"] * df["backlinks"]
    df["content_speed_ratio"] = df["content_length"] / (df["page_speed"] + 0.1)
    return df

# ==============================
# src/model.py
# ==============================
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.config import MODEL_PATH

def train(df):
    X = df.drop("ranking", axis=1)
    y = df["ranking"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = XGBRegressor(n_estimators=200, max_depth=6)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print("RMSE:", rmse)
    print("R2:", r2)

    joblib.dump(model, MODEL_PATH)
    return model

# ==============================
# src/explain.py
# ==============================
import shap
import joblib
from src.config import MODEL_PATH

def explain(df):
    model = joblib.load(MODEL_PATH)
    explainer = shap.Explainer(model)
    shap_values = explainer(df.drop("ranking", axis=1))
    shap.plots.bar(shap_values)

# ==============================
# app/api.py
# ==============================
from fastapi import FastAPI
import joblib
import pandas as pd
from src.config import MODEL_PATH

app = FastAPI()
model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"status": "SEO Ranking Predictor API"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"predicted_ranking": float(pred)}

# ==============================
# app/dashboard.py
# ==============================
import streamlit as st
import joblib
import pandas as pd
from src.config import MODEL_PATH

model = joblib.load(MODEL_PATH)

st.title("SEO Ranking Predictor")

backlinks = st.slider("Backlinks", 0, 5000)
ref_domains = st.slider("Ref Domains", 0, 500)
content_length = st.slider("Content Length", 300, 5000)
page_speed = st.slider("Page Speed", 0.5, 5.0)
domain_authority = st.slider("Domain Authority", 1, 100)
internal_links = st.slider("Internal Links", 0, 100)
external_links = st.slider("External Links", 0, 50)

if st.button("Predict"):
    df = pd.DataFrame([{
        "backlinks": backlinks,
        "ref_domains": ref_domains,
        "content_length": content_length,
        "page_speed": page_speed,
        "domain_authority": domain_authority,
        "internal_links": internal_links,
        "external_links": external_links
    }])

    pred = model.predict(df)[0]
    st.success(f"Predicted Ranking: {round(pred, 2)}")

# ==============================
# main.py (Pipeline Runner)
# ==============================
from src.data_loader import generate_data
from src.feature_engineering import engineer
from src.model import train

print("Generating data...")
df = generate_data()

print("Engineering features...")
df = engineer(df)

print("Training model...")
train(df)

print("Done. Run API or Dashboard.")

# ==============================
# Dockerfile
# ==============================
# FROM python:3.10
# WORKDIR /app
# COPY . .
# RUN pip install -r requirements.txt
# CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ==============================
# README.md (Summary)
# ==============================
# SEO Ranking Predictor
# - Predict keyword ranking using ML
# - Features: backlinks, content, speed
# - API + Dashboard included

# Run:
# python main.py
# streamlit run app/dashboard.py
# uvicorn app.api:app --reload
