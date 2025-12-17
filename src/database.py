from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, String, Text, DateTime, Boolean, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.sqlite import JSON

from src.config import settings

Base = declarative_base()

class UserModel(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String)
    # Profile fields
    full_name = Column(String)
    bio = Column(Text)
    phone = Column(String)
    avatar_url = Column(String)

class ConversationModel(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    title = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class MessageModel(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, index=True)
    conversation_id = Column(String, index=True)
    user_id = Column(String, index=True)
    content = Column(Text)
    role = Column(String)  # 'user', 'assistant', 'system'
    model_used = Column(String)
    tokens_used = Column(Integer)
    response_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    extra_metadata = Column(JSON)

class UniversityDataModel(Base):
    __tablename__ = "university_data"
    
    id = Column(String, primary_key=True, index=True)
    url = Column(String)
    title = Column(String)
    content = Column(Text)
    content_type = Column(String)  # 'fee_structure', 'admission', 'course_info', etc.
    scraped_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    extra_metadata = Column(JSON)

class DocumentChunkModel(Base):
    __tablename__ = "document_chunks"
    
    id = Column(String, primary_key=True, index=True)
    university_data_id = Column(String, index=True)
    chunk_text = Column(Text)
    chunk_index = Column(Integer)
    vector_id = Column(String)  # Reference to vector in FAISS/ChromaDB
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLogModel(Base):
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    action = Column(String)
    resource = Column(String)
    details = Column(JSON)
    ip_address = Column(String)
    user_agent = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self):
        self.engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            echo=settings.LOG_LEVEL == "DEBUG"
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()
    
    def create_tables(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def create_user(self, user_data: dict) -> UserModel:
        """Create a new user in the database."""
        with self.get_session() as db:
            db_user = UserModel(**user_data)
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            return db_user
    
    def get_user_by_username(self, username: str) -> Optional[UserModel]:
        """Get user by username."""
        with self.get_session() as db:
            return db.query(UserModel).filter(UserModel.username == username).first()
    
    def get_user_by_email(self, email: str) -> Optional[UserModel]:
        """Get user by email."""
        with self.get_session() as db:
            return db.query(UserModel).filter(UserModel.email == email).first()
    
    def get_user_by_id(self, user_id: str) -> Optional[UserModel]:
        """Get user by ID."""
        with self.get_session() as db:
            return db.query(UserModel).filter(UserModel.id == user_id).first()
    
    def get_all_users(self) -> List[UserModel]:
        """Get all users."""
        with self.get_session() as db:
            return db.query(UserModel).all()
    
    def update_user_last_login(self, user_id: str, last_login: datetime):
        """Update user's last login timestamp."""
        with self.get_session() as db:
            user = db.query(UserModel).filter(UserModel.id == user_id).first()
            if user:
                user.last_login = last_login
                user.updated_at = datetime.utcnow()
                db.commit()
    
    def update_user_password(self, user_id: str, hashed_password: str):
        """Update user's password."""
        with self.get_session() as db:
            user = db.query(UserModel).filter(UserModel.id == user_id).first()
            if user:
                user.hashed_password = hashed_password
                user.updated_at = datetime.utcnow()
                db.commit()
    
    def update_user_profile(self, user_id: str, profile: dict) -> Optional[UserModel]:
        """Update user profile fields."""
        with self.get_session() as db:
            user = db.query(UserModel).filter(UserModel.id == user_id).first()
            if not user:
                return None
            for key in ["full_name", "bio", "phone", "avatar_url", "is_active"]:
                if key in profile:
                    setattr(user, key, profile[key])
            user.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(user)
            return user
    
    def create_conversation(self, conversation_data: dict) -> ConversationModel:
        """Create a new conversation."""
        with self.get_session() as db:
            db_conversation = ConversationModel(**conversation_data)
            db.add(db_conversation)
            db.commit()
            db.refresh(db_conversation)
            return db_conversation
    
    def get_user_conversations(self, user_id: str, limit: int = 50) -> List[ConversationModel]:
        """Get user's conversations."""
        with self.get_session() as db:
            return db.query(ConversationModel).filter(
                ConversationModel.user_id == user_id,
                ConversationModel.is_active == True
            ).order_by(ConversationModel.updated_at.desc()).limit(limit).all()
    
    def create_message(self, message_data: dict) -> MessageModel:
        """Create a new message."""
        with self.get_session() as db:
            db_message = MessageModel(**message_data)
            db.add(db_message)
            db.commit()
            db.refresh(db_message)
            return db_message
    
    def get_conversation_messages(self, conversation_id: str, limit: int = 100) -> List[MessageModel]:
        """Get messages from a conversation."""
        with self.get_session() as db:
            return db.query(MessageModel).filter(
                MessageModel.conversation_id == conversation_id
            ).order_by(MessageModel.created_at).limit(limit).all()
    
    def store_university_data(self, data: dict) -> UniversityDataModel:
        """Store scraped university data."""
        with self.get_session() as db:
            db_data = UniversityDataModel(**data)
            db.add(db_data)
            db.commit()
            db.refresh(db_data)
            return db_data
    
    def get_university_data(self, content_type: Optional[str] = None) -> List[UniversityDataModel]:
        """Get university data, optionally filtered by content type."""
        with self.get_session() as db:
            query = db.query(UniversityDataModel).filter(UniversityDataModel.is_active == True)
            if content_type:
                query = query.filter(UniversityDataModel.content_type == content_type)
            return query.order_by(UniversityDataModel.scraped_at.desc()).all()
    
    def store_document_chunk(self, chunk_data: dict) -> DocumentChunkModel:
        """Store a document chunk."""
        with self.get_session() as db:
            db_chunk = DocumentChunkModel(**chunk_data)
            db.add(db_chunk)
            db.commit()
            db.refresh(db_chunk)
            return db_chunk
    
    def get_document_chunks(self, university_data_id: str) -> List[DocumentChunkModel]:
        """Get chunks for a university data item."""
        with self.get_session() as db:
            return db.query(DocumentChunkModel).filter(
                DocumentChunkModel.university_data_id == university_data_id
            ).order_by(DocumentChunkModel.chunk_index).all()
    
    def log_audit_event(self, audit_data: dict):
        """Log an audit event."""
        with self.get_session() as db:
            audit_log = AuditLogModel(**audit_data)
            db.add(audit_log)
            db.commit()
    
    def get_audit_logs(self, user_id: Optional[str] = None, limit: int = 100) -> List[AuditLogModel]:
        """Get audit logs."""
        with self.get_session() as db:
            query = db.query(AuditLogModel)
            if user_id:
                query = query.filter(AuditLogModel.user_id == user_id)
            return query.order_by(AuditLogModel.timestamp.desc()).limit(limit).all()
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user account and all associated data."""
        with self.get_session() as db:
            try:
                # Check if user exists
                user = db.query(UserModel).filter(UserModel.id == user_id).first()
                if not user:
                    return False
                
                # Delete associated data
                # 1. Delete user's conversations
                conversations = db.query(ConversationModel).filter(ConversationModel.user_id == user_id).all()
                conversation_ids = [conv.id for conv in conversations]
                
                # 2. Delete messages from user's conversations
                for conv_id in conversation_ids:
                    db.query(MessageModel).filter(MessageModel.conversation_id == conv_id).delete()
                
                # 3. Delete conversations
                db.query(ConversationModel).filter(ConversationModel.user_id == user_id).delete()
                
                # 4. Delete audit logs
                db.query(AuditLogModel).filter(AuditLogModel.user_id == user_id).delete()
                
                # 5. Delete the user account
                db.delete(user)
                
                # Commit all changes
                db.commit()
                
                return True
                
            except Exception as e:
                db.rollback()
                raise e

# Global database manager instance
db_manager = DatabaseManager()
