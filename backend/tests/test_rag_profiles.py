import pytest
from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)

def test_rajat_sharma_two_turn():
    # First query
    response1 = client.post(\"/chat\", json={\"message\": \"who is rajat sharma\"})
    assert response1.status_code == 200
    j1 = response1.json()
    assert 'reply' in j1
    assert 'data' in j1
    assert 'Dr.' in j1['reply'] or 'Rajat Sharma' in j1['reply']

    session_id = 'test_session_123'

    # Second query (follow-up)
    response2 = client.post(\"/chat\", json={\"message\": \"give me more details about him\", \"session_id\": session_id})
    assert response2.status_code == 200
    j2 = response2.json()
    data = j2.get('data', {})
    assert data.get('education') and len(data['education']) > 0
    assert any('Ph.D.' in e or 'Computational Neuroscience' in e for e in data['education'])
    assert data.get('research') and len(data['research']) > 0
    assert data.get('links') and len(data['links']) > 0
    assert data.get('sources') and len(data['sources']) > 0
    assert 'Not available in sources' not in j2['reply'] or 'Ph.D.' in j2['reply']
