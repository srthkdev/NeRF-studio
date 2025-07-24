#!/usr/bin/env python3
"""
Startup script for NeRF Studio Backend
"""

import uvicorn
from app.main import app
from app.core.config import settings

if __name__ == "__main__":
    print("🚀 Starting NeRF Studio Backend...")
    print(f"📊 API Documentation: http://localhost:8000/docs")
    print(f"🔧 ReDoc Documentation: http://localhost:8000/redoc")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 