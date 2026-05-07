import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def client():
    # Mock the RAG chain so API tests don't need full pipeline loaded
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "result": "The turbine engine hot section requires visual inspection of blades and combustion liners.",
        "source_documents": [
            MagicMock(metadata={"source": "faa_powerplant", "page": 415},
                      page_content="Turbine inspection content...")
        ]
    }
    with patch("src.api.main.rag_chain", mock_chain):
        from src.api.main import app
        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "AeroManual" in response.json()["message"]


def test_query_endpoint_returns_answer(client):
    response = client.post("/query", json={
        "question": "What are the inspection requirements for turbine engines?",
        "top_k": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 10


def test_query_endpoint_returns_sources(client):
    response = client.post("/query", json={
        "question": "Explain fuel system safety precautions",
        "top_k": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert "sources" in data
    assert "num_sources" in data


def test_query_empty_question(client):
    response = client.post("/query", json={"question": "", "top_k": 5})
    # Should still return 200 or handle gracefully
    assert response.status_code in [200, 422]


def test_health_shows_pipeline_status(client):
    response = client.get("/health")
    data = response.json()
    assert "pipeline_loaded" in data