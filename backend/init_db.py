#!/usr/bin/env python3
"""
Database initialization script for NeRF Studio Backend
Creates the SQLite database and tables
"""

import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from app.database import Base
from app.core.config import settings
# Import models to register them with SQLAlchemy
from app.models import Project, TrainingJob

async def init_db():
    """Initialize the database with all tables"""
    print("Initializing database...")
    
    # Create engine
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=True,
        future=True,
        connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
    )
    
    try:
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Database initialized successfully!")
        print(f"Database file: {settings.DATABASE_URL.replace('sqlite+aiosqlite:///', '')}")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        raise
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(init_db()) 