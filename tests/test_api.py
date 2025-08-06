import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_valid_prediction():
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert type(response.json()["prediction"]) == int

def test_invalid_shape():
    response = client.post("/predict", json={"features": [1.0, 2.0]})  # too short
    assert response.status_code == 400
    assert response.json()["detail"] == "Exactly 4 features are required."

def test_empty_input():
    response = client.post("/predict", json={"features": []})
    assert response.status_code == 400
    assert response.json()["detail"] == "Exactly 4 features are required."
