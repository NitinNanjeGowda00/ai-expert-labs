from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(
    title="AI Engineering Labs – Breast Cancer Classifier",
    description="Logistic Regression model deployed with FastAPI + Docker",
    version="1.0.0",
)

# Load model
MODEL_PATH = Path("models/logreg_model.joblib")
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    prediction: int
    probability: float


@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "logistic_regression",
        "task": "binary_classification",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    X = np.array(req.features).reshape(1, -1)

    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    return PredictResponse(prediction=pred, probability=prob)
