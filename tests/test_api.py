import pytest
from fastapi.testclient import TestClient
from src.app import app

@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager runs FastAPI startup/shutdown events.
    with TestClient(app) as test_client:
        yield test_client

def test_home_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_setosa(client):
    response = client.post("/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    assert response.json()["prediction"] == "setosa"

def test_invalid_input_is_rejected(client):
    response = client.post("/predict", json={
        "sepal_length": -1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 422
