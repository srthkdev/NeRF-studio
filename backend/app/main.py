
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import api as api_v1
from app.database import engine
from app.models import Base
from app.core.config import settings
from app.core.performance_monitor import start_global_monitoring, stop_global_monitoring
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
# Reduce SQLAlchemy logging verbosity
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for NeRF Studio, a platform for creating 3D scenes from 2D images.",
    version=settings.VERSION,
)

# Set up CORS middleware
if settings.DEBUG:
    # In debug mode, allow all origins for easier development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # Can't use credentials with allow_origins=["*"]
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # In production, use specific origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.on_event("startup")
async def startup_event():
    # Create tables if they don't exist (for development, use Alembic for migrations in production)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    start_global_monitoring()

@app.on_event("shutdown")
async def shutdown_event():
    await engine.dispose()
    stop_global_monitoring()

# Include the v1 API router
app.include_router(api_v1.router, prefix=settings.API_V1_STR, tags=["v1"])

@app.get("/")
def read_root():
    return {"message": "Welcome to NeRF Studio API"}
