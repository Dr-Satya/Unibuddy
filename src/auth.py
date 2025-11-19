import hashlib
import secrets
import time
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

# Fix for bcrypt 4.x compatibility with passlib
try:
    import bcrypt
    if not hasattr(bcrypt, '__about__'):
        # Create a mock __about__ object for bcrypt 4.x compatibility
        class MockAbout:
            __version__ = bcrypt.__version__
        bcrypt.__about__ = MockAbout()
except ImportError:
    pass

from passlib.context import CryptContext
from jose import JWTError, jwt
from email_validator import validate_email, EmailNotValidError
import secrets

from src.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@dataclass
class User:
    id: str
    username: str
    email: str
    hashed_password: str
    is_active: bool = True
    created_at: datetime = None
    last_login: datetime = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    is_active: bool = True
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class AuthManager:
    """Handles authentication, user management, and session control."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.failed_attempts: Dict[str, int] = {}
        
    def hash_password(self, password: str) -> str:
        """Hash a password for storing."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a stored password against one provided by user."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            return payload
        except JWTError:
            return None
    
    def create_user(self, username: str, email: str, password: str) -> User:
        """Create a new user."""
        user_id = hashlib.sha256(f"{username}{email}{time.time()}".encode()).hexdigest()
        hashed_password = self.hash_password(password)
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            hashed_password=hashed_password
        )
        
        self.users[user_id] = user
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password."""
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            return None
            
        # Check for too many failed attempts
        if self.failed_attempts.get(username, 0) >= 5:
            return None
            
        if not self.verify_password(password, user.hashed_password):
            self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
            return None
        
        # Reset failed attempts on successful login
        if username in self.failed_attempts:
            del self.failed_attempts[username]
        
        user.last_login = datetime.now()
        return user
    
    def create_session(self, user_id: str, ip_address: Optional[str] = None, 
                      user_agent: Optional[str] = None) -> Session:
        """Create a new session for a user."""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=24)
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        session = self.sessions.get(session_id)
        if not session:
            return None
            
        if not session.is_active or datetime.now() > session.expires_at:
            return None
            
        return session
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            return True
        return False
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        return self.users.get(user_id)
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time > session.expires_at
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
    
    def validate_email(self, email: str) -> Tuple[bool, str]:
        """Validate email format and domain."""
        try:
            validated_email = validate_email(email)
            return True, validated_email.email
        except EmailNotValidError as e:
            return False, str(e)
    
    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength and return list of issues."""
        issues = []
        
        # Check minimum length
        if len(password) < 8:
            issues.append("Password must be at least 8 characters long")
        
        # Check for uppercase letter
        if not re.search(r'[A-Z]', password):
            issues.append("Password must contain at least one uppercase letter")
        
        # Check for lowercase letter
        if not re.search(r'[a-z]', password):
            issues.append("Password must contain at least one lowercase letter")
        
        # Check for digit
        if not re.search(r'\d', password):
            issues.append("Password must contain at least one digit")
        
        # Check for special character
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            issues.append("Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)")
        
        # Check for common patterns
        common_patterns = ['password', '123456', 'qwerty', 'abc123', 'password123']
        if password.lower() in [p.lower() for p in common_patterns]:
            issues.append("Password cannot be a common pattern")
        
        return len(issues) == 0, issues
    
    def validate_username(self, username: str) -> Tuple[bool, str]:
        """Validate username format."""
        if len(username) < 3:
            return False, "Username must be at least 3 characters long"
        
        if len(username) > 30:
            return False, "Username must be no more than 30 characters long"
        
        if not re.match(r'^[a-zA-Z0-9_.-]+$', username):
            return False, "Username can only contain letters, numbers, dots, hyphens, and underscores"
        
        if username.startswith('.') or username.endswith('.'):
            return False, "Username cannot start or end with a dot"
        
        return True, "Valid username"
    
    def check_username_availability(self, username: str) -> bool:
        """Check if username is available (not in memory and not in database)."""
        # Check in-memory users
        for user in self.users.values():
            if user.username.lower() == username.lower():
                return False
        
        # Check in database
        try:
            from src.database import db_manager
            db_user = db_manager.get_user_by_username(username)
            return db_user is None
        except Exception:
            return True  # If DB check fails, allow username
    
    def check_email_availability(self, email: str) -> bool:
        """Check if email is available (not in memory and not in database)."""
        # Check in-memory users
        for user in self.users.values():
            if user.email.lower() == email.lower():
                return False
        
        # Check in database
        try:
            from src.database import db_manager
            db_user = db_manager.get_user_by_email(email)
            return db_user is None
        except Exception:
            return True  # If DB check fails, allow email
    
    def register_user(self, username: str, email: str, password: str, 
                     confirm_password: str) -> Tuple[bool, str, Optional[User]]:
        """Register a new user with comprehensive validation."""
        # Validate password confirmation
        if password != confirm_password:
            return False, "Passwords do not match", None
        
        # Validate username
        username_valid, username_msg = self.validate_username(username)
        if not username_valid:
            return False, username_msg, None
        
        # Check username availability
        if not self.check_username_availability(username):
            return False, "Username already exists", None
        
        # Validate email
        email_valid, email_result = self.validate_email(email)
        if not email_valid:
            return False, f"Invalid email: {email_result}", None
        email = email_result  # Use validated/normalized email
        
        # Check email availability
        if not self.check_email_availability(email):
            return False, "Email already registered", None
        
        # Validate password strength
        password_valid, password_issues = self.validate_password_strength(password)
        if not password_valid:
            return False, "Password requirements not met:\n• " + "\n• ".join(password_issues), None
        
        try:
            # Create user in memory
            user = self.create_user(username, email, password)
            
            # Store in database
            from src.database import db_manager
            user_data = {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'hashed_password': user.hashed_password,
                'is_active': user.is_active,
                'created_at': user.created_at,
                'mfa_enabled': user.mfa_enabled,
                'mfa_secret': user.mfa_secret
            }
            
            db_manager.create_user(user_data)
            
            return True, "User registered successfully!", user
            
        except Exception as e:
            # Remove from memory if DB creation failed
            if user.id in self.users:
                del self.users[user.id]
            return False, f"Registration failed: {str(e)}", None
    
    def authenticate_user_enhanced(self, username: str, password: str) -> Tuple[bool, str, Optional[User]]:
        """Enhanced authentication with better error messages."""
        # Check if account is locked
        if self.failed_attempts.get(username, 0) >= 5:
            return False, "Account temporarily locked due to multiple failed attempts. Please try again later.", None
        
        # First check in-memory users
        user = None
        for u in self.users.values():
            if u.username.lower() == username.lower():
                user = u
                break
        
        # If not found in memory, check database
        if not user:
            try:
                from src.database import db_manager
                db_user = db_manager.get_user_by_username(username)
                if db_user:
                    # Load user into memory
                    user = User(
                        id=db_user.id,
                        username=db_user.username,
                        email=db_user.email,
                        hashed_password=db_user.hashed_password,
                        is_active=db_user.is_active,
                        created_at=db_user.created_at,
                        last_login=db_user.last_login,
                        mfa_enabled=db_user.mfa_enabled,
                        mfa_secret=db_user.mfa_secret
                    )
                    self.users[user.id] = user
            except Exception as e:
                return False, f"Database error during authentication: {str(e)}", None
        
        if not user:
            self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
            return False, "Invalid username or password", None
        
        if not user.is_active:
            return False, "Account is disabled", None
        
        # Verify password
        if not self.verify_password(password, user.hashed_password):
            self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
            return False, "Invalid username or password", None
        
        # Reset failed attempts on successful login
        if username in self.failed_attempts:
            del self.failed_attempts[username]
        
        # Update last login
        user.last_login = datetime.now()
        
        # Update in database
        try:
            from src.database import db_manager
            db_manager.update_user_last_login(user.id, user.last_login)
        except Exception:
            pass  # Non-critical error
        
        return True, "Authentication successful", user
    
    def get_failed_attempts(self, username: str) -> int:
        """Get number of failed login attempts for a username."""
        return self.failed_attempts.get(username, 0)
    
    def reset_failed_attempts(self, username: str):
        """Reset failed login attempts for a username."""
        if username in self.failed_attempts:
            del self.failed_attempts[username]
    
    def change_password(self, user_id: str, current_password: str, new_password: str, 
                      confirm_password: str) -> Tuple[bool, str]:
        """Change user password."""
        user = self.get_user(user_id)
        if not user:
            return False, "User not found"
        
        # Verify current password
        if not self.verify_password(current_password, user.hashed_password):
            return False, "Current password is incorrect"
        
        # Validate new password
        if new_password != confirm_password:
            return False, "New passwords do not match"
        
        password_valid, password_issues = self.validate_password_strength(new_password)
        if not password_valid:
            return False, "New password requirements not met:\n• " + "\n• ".join(password_issues)
        
        # Don't allow same password
        if self.verify_password(new_password, user.hashed_password):
            return False, "New password must be different from current password"
        
        try:
            # Update password
            user.hashed_password = self.hash_password(new_password)
            
            # Update in database
            from src.database import db_manager
            db_manager.update_user_password(user_id, user.hashed_password)
            
            return True, "Password updated successfully"
            
        except Exception as e:
            return False, f"Failed to update password: {str(e)}"
    
    def generate_password_reset_token(self, user_id: str, expires_minutes: int = 30) -> str:
        """Generate a password reset token."""
        data = {"sub": user_id, "type": "password_reset"}
        return self.create_access_token(data, timedelta(minutes=expires_minutes))
    
    def reset_password_with_token(self, token: str, new_password: str, confirm_password: str) -> Tuple[bool, str]:
        """Reset password using a valid token."""
        payload = self.verify_token(token)
        if not payload or payload.get("type") != "password_reset":
            return False, "Invalid or expired token"
        user_id = payload.get("sub")
        user = self.get_user(user_id)
        if not user:
            # Try database lookup
            try:
                from src.database import db_manager
                db_user = db_manager.get_user_by_id(user_id)
                if not db_user:
                    return False, "User not found"
                user = User(
                    id=db_user.id,
                    username=db_user.username,
                    email=db_user.email,
                    hashed_password=db_user.hashed_password,
                    is_active=db_user.is_active,
                    created_at=db_user.created_at,
                )
                self.users[user.id] = user
            except Exception:
                return False, "User lookup failed"
        # Validate and set password
        if new_password != confirm_password:
            return False, "Passwords do not match"
        valid, issues = self.validate_password_strength(new_password)
        if not valid:
            return False, "Password requirements not met:\n• " + "\n• ".join(issues)
        user.hashed_password = self.hash_password(new_password)
        try:
            from src.database import db_manager
            db_manager.update_user_password(user.id, user.hashed_password)
            return True, "Password reset successfully"
        except Exception as e:
            return False, f"Failed to reset password: {str(e)}"
    
    def load_users_from_database(self):
        """Load all users from database into memory for faster access."""
        try:
            from src.database import db_manager
            db_users = db_manager.get_all_users()
            
            for db_user in db_users:
                user = User(
                    id=db_user.id,
                    username=db_user.username,
                    email=db_user.email,
                    hashed_password=db_user.hashed_password,
                    is_active=db_user.is_active,
                    created_at=db_user.created_at,
                    last_login=db_user.last_login,
                    mfa_enabled=db_user.mfa_enabled,
                    mfa_secret=db_user.mfa_secret
                )
                self.users[user.id] = user
            
            print(f"Loaded {len(db_users)} users from database")
            
        except Exception as e:
            print(f"Warning: Could not load users from database: {e}")
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username from memory or database."""
        # Check memory first
        for user in self.users.values():
            if user.username.lower() == username.lower():
                return user
        
        # Check database
        try:
            from src.database import db_manager
            db_user = db_manager.get_user_by_username(username)
            if db_user:
                user = User(
                    id=db_user.id,
                    username=db_user.username,
                    email=db_user.email,
                    hashed_password=db_user.hashed_password,
                    is_active=db_user.is_active,
                    created_at=db_user.created_at,
                    last_login=db_user.last_login,
                    mfa_enabled=db_user.mfa_enabled,
                    mfa_secret=db_user.mfa_secret
                )
                self.users[user.id] = user
                return user
        except Exception:
            pass
        
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email from memory or database."""
        # Check memory first
        for user in self.users.values():
            if user.email.lower() == email.lower():
                return user
        
        # Check database
        try:
            from src.database import db_manager
            db_user = db_manager.get_user_by_email(email)
            if db_user:
                user = User(
                    id=db_user.id,
                    username=db_user.username,
                    email=db_user.email,
                    hashed_password=db_user.hashed_password,
                    is_active=db_user.is_active,
                    created_at=db_user.created_at,
                    last_login=db_user.last_login,
                    mfa_enabled=db_user.mfa_enabled,
                    mfa_secret=db_user.mfa_secret
                )
                self.users[user.id] = user
                return user
        except Exception:
            pass
        
        return None
    
    def delete_user_account(self, user_id: str, current_password: str) -> Tuple[bool, str]:
        """Delete a user account with password verification."""
        user = self.get_user(user_id)
        if not user:
            # Try to get from database
            try:
                from src.database import db_manager
                db_user = db_manager.get_user_by_id(user_id)
                if not db_user:
                    return False, "User not found"
                user = User(
                    id=db_user.id,
                    username=db_user.username,
                    email=db_user.email,
                    hashed_password=db_user.hashed_password,
                    is_active=db_user.is_active,
                    created_at=db_user.created_at,
                    last_login=db_user.last_login,
                    mfa_enabled=db_user.mfa_enabled,
                    mfa_secret=db_user.mfa_secret
                )
            except Exception as e:
                return False, f"Error verifying user: {str(e)}"
        
        # Verify current password
        if not self.verify_password(current_password, user.hashed_password):
            return False, "Current password is incorrect"
        
        try:
            # Remove from memory
            if user.id in self.users:
                del self.users[user.id]
            
            # Clear any failed attempts for this user
            if user.username in self.failed_attempts:
                del self.failed_attempts[user.username]
            
            # Invalidate all sessions for this user
            sessions_to_remove = []
            for session_id, session in self.sessions.items():
                if session.user_id == user.id:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.sessions[session_id]
            
            # Delete from database
            from src.database import db_manager
            success = db_manager.delete_user(user_id)
            
            if success:
                return True, f"Account '{user.username}' has been permanently deleted along with all associated data"
            else:
                return False, "Failed to delete account from database"
                
        except Exception as e:
            return False, f"Failed to delete account: {str(e)}"

# Global auth manager instance
auth_manager = AuthManager()
