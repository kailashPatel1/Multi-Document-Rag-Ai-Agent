import json
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey, Boolean, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from ..config.settings import settings

Base = declarative_base()
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class DocumentModel(Base):
    __tablename__ = "documents"
    id = Column(String(64), primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_name = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, default=0)
    total_pages = Column(Integer, default=1)
    total_chunks = Column(Integer, default=0)
    summary = Column(Text, nullable=True)
    columns_json = Column(Text, nullable=True)  # for CSV
    status = Column(String(30), default="ready")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    chunks = relationship("DocumentChunkModel", back_populates="document", cascade="all, delete-orphan")

class DocumentChunkModel(Base):
    __tablename__ = "document_chunks"
    id = Column(String(128), primary_key=True, index=True)
    document_id = Column(String(64), ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    document_name = Column(String(255), nullable=False)
    page_number = Column(Integer, default=1)
    chunk_index = Column(Integer, default=0)
    content = Column(Text, nullable=False)
    char_count = Column(Integer, default=0)
    metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    document = relationship("DocumentModel", back_populates="chunks")

class ChatSessionModel(Base):
    __tablename__ = "chat_sessions"
    id = Column(String(64), primary_key=True, index=True)
    title = Column(String(255), default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    messages = relationship("ChatMessageModel", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessageModel.created_at")

class ChatMessageModel(Base):
    __tablename__ = "chat_messages"
    id = Column(String(64), primary_key=True, index=True)
    session_id = Column(String(64), ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role = Column(String(20), nullable=False)  # user, assistant, system, tool
    content = Column(Text, nullable=False)
    citations_json = Column(Text, default="[]")
    tool_traces_json = Column(Text, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    session = relationship("ChatSessionModel", back_populates="messages")

class EvaluationRunModel(Base):
    __tablename__ = "evaluation_runs"
    id = Column(String(64), primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    total_queries = Column(Integer, default=0)
    retrieval_precision = Column(Float, default=0.0)
    retrieval_recall = Column(Float, default=0.0)
    context_relevance = Column(Float, default=0.0)
    answer_faithfulness = Column(Float, default=0.0)
    answer_correctness = Column(Float, default=0.0)
    citation_accuracy = Column(Float, default=0.0)
    average_latency_ms = Column(Float, default=0.0)
    details_json = Column(Text, default="[]")

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
