from fastapi import APIRouter

api_router = APIRouter()


@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "nerf-studio-backend"}


@api_router.get("/projects")
async def list_projects():
    """List all NeRF projects"""
    return {"projects": []}


@api_router.post("/projects")
async def create_project():
    """Create a new NeRF project"""
    return {"message": "Project creation endpoint - to be implemented"}


@api_router.post("/upload")
async def upload_images():
    """Upload images for NeRF training"""
    return {"message": "Image upload endpoint - to be implemented"}