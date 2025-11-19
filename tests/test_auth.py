import os
import tempfile
import pytest
from datetime import datetime

# Ensure settings points to a temp sqlite for tests
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from src.auth import auth_manager
from src.database import db_manager

def test_registration_and_login_flow():
    username = "testuser"
    email = "testuser@example.com"
    password = "Str0ng!Pass"

    ok, msg, user = auth_manager.register_user(username, email, password, password)
    assert ok, f"Registration failed: {msg}"
    assert user is not None

    # Duplicate username
    ok2, msg2, user2 = auth_manager.register_user(username, "other@example.com", password, password)
    assert not ok2 and "Username" in msg2

    # Duplicate email
    ok3, msg3, user3 = auth_manager.register_user("otheruser", email, password, password)
    assert not ok3 and "Email" in msg3

    # Login success
    success, lmsg, luser = auth_manager.authenticate_user_enhanced(username, password)
    assert success, f"Login failed: {lmsg}"
    assert luser is not None

    # Wrong password attempts
    success2, lmsg2, _ = auth_manager.authenticate_user_enhanced(username, "wrong")
    assert not success2


def test_change_password_and_profile_update():
    username = "profileuser"
    email = "profile@example.com"
    password = "Str0ng!Pass2"

    ok, msg, user = auth_manager.register_user(username, email, password, password)
    assert ok

    # Change password
    ok2, msg2 = auth_manager.change_password(user.id, password, "N3w!Passw0rd", "N3w!Passw0rd")
    assert ok2, msg2

    # Login with new password
    success, lmsg, _ = auth_manager.authenticate_user_enhanced(username, "N3w!Passw0rd")
    assert success, lmsg

    # Update profile
    updated = db_manager.update_user_profile(user.id, {
        "full_name": "Test User",
        "phone": "+1-555-0100",
        "bio": "Hello",
        "avatar_url": "http://example.com/avatar.png"
    })
    assert updated is not None
    assert updated.full_name == "Test User"

