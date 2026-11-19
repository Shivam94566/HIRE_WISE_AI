import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Fix path to allow importing serve module directly
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from serve import create_app
except ImportError:
    from mlops.serve import create_app

class FakeModel:
    """
    Fake model to simulate predictions without needing a live MLflow server.
    """
    def predict(self, features):
        # Feature format: [[claimed, measured, difference]]
        results = []
        for row in features:
            diff = row[2]
            results.append(1 if diff > 2.0 else 0)
        return results

@pytest.fixture
def client():
    # Application factory ke zariye FakeModel inject karke isolated TestClient banate hain
    test_app = create_app(model_loader=lambda: FakeModel())
    with TestClient(test_app) as test_client:
        yield test_client

def test_health_endpoint(client):
    """Verify /health returns 200 and status ok"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_inflated_resume(client):
    """Verify inflated resume (gap > 2.0) returns inflated=True"""
    payload = {"claimed_skill": 9.0, "measured_ability": 4.5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["inflated"] is True
    assert data["difference"] == 4.5

def test_predict_honest_resume(client):
    """Verify honest resume (gap <= 2.0) returns inflated=False"""
    payload = {"claimed_skill": 6.5, "measured_ability": 6.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["inflated"] is False
    assert data["difference"] == 0.5

def test_predict_invalid_missing_fields(client):
    """Verify 422 Unprocessable Entity when required field is missing"""
    response = client.post("/predict", json={"claimed_skill": 8.0})
    assert response.status_code == 422

def test_predict_invalid_field_types(client):
    """Verify 422 Unprocessable Entity when fields have invalid types"""
    response = client.post("/predict", json={"claimed_skill": "expert", "measured_ability": 5.0})
    assert response.status_code == 422
