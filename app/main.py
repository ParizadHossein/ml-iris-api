# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

with open("model/iris_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class IrisInput(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: IrisInput):
    if len(data.features) != 4:
        raise HTTPException(status_code=400, detail="Exactly 4 features are required.")

    prediction = model.predict([data.features])
    return {"prediction": int(prediction[0])}