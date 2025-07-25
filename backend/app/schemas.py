
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class ProjectBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class ProjectUpdate(ProjectBase):
    pass

class ProjectInDB(ProjectBase):
    id: str
    created_at: datetime
    updated_at: datetime
    data: Dict[str, Any]
    config: Dict[str, Any]
    checkpoints: Dict[str, Any]

    class Config:
        from_attributes = True

class TrainingJobBase(BaseModel):
    project_id: str

class TrainingJobCreate(TrainingJobBase):
    pass

class TrainingJobInDB(TrainingJobBase):
    id: str
    status: str
    progress: float
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Dict[str, Any]

    class Config:
        from_attributes = True
