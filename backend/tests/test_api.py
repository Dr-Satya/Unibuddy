import os
import time
import requests
from src.api_adapter import get_reply

BASE_URL = os.environ.get('UNIBUDDY_API_URL', 'http://127.0.0.1:9000')
API_KEY = os.environ.get('UNIBUDDY_API_KEY')


def test_get_reply_local():
    # quick unit smoke test for adapter
    r = get_reply('What is the fee for B.Tech Computer Science?')
    assert isinstance(r, dict)
    assert 'reply' in r
    print('adapter reply length:', len(r.get('reply','')))


def test_api_chat_endpoint():
    assert API_KEY, 'Set UNIBUDDY_API_KEY env var before running this test'
    url = f"{BASE_URL}/chat"
    payload = {"message": "Hello, test message"}
    headers = {'X-API-Key': API_KEY, 'Content-Type': 'application/json'}
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert 'reply' in data
    print('api reply length:', len(data.get('reply','')))
