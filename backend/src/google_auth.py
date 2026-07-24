from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import RedirectResponse
from urllib.parse import urlencode

from src.config import settings

router = APIRouter()

GOOGLE_CLIENT_ID = getattr(settings, 'GOOGLE_CLIENT_ID', None)
GOOGLE_CLIENT_SECRET = getattr(settings, 'GOOGLE_CLIENT_SECRET', None)
REDIRECT_URI = settings.GOOGLE_REDIRECT_URI

@router.get('/auth/google/login')
def google_login():
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail='Google client not configured')
    params = {
        'client_id': GOOGLE_CLIENT_ID,
        'response_type': 'code',
        'scope': 'openid email profile',
        'redirect_uri': REDIRECT_URI,
        'access_type': 'offline',
        'prompt': 'consent'
    }
    url = 'https://accounts.google.com/o/oauth2/v2/auth?' + urlencode(params)
    return RedirectResponse(url)

@router.get('/auth/google/callback')
def google_callback(request: Request):
    code = request.query_params.get('code')
    if not code:
        raise HTTPException(status_code=400, detail='Missing code')
    # In production exchange code for token here (omitted for tests)
    return {'status': 'ok', 'code': code}
