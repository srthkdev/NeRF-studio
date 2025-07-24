
from sqlalchemy import Column, String, DateTime, JSON, Text, Float
import uuid
from datetime import datetime
from app.database import Base

class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    data = Column(JSON, default=dict)
    config = Column(JSON, default=dict)
    checkpoints = Column(JSON, default=dict)

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(String, primary_key=True, nullable=False)
    project_id = Column(String, nullable=False)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    metrics = Column(JSON, default=dict)
    progress = Column(Float, default=0.0)
