from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models import Project, TrainingJob
from app.schemas import ProjectCreate, ProjectUpdate, TrainingJobCreate
from uuid import UUID, uuid4
import os
import shutil
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class ProjectService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_project(self, project_in: ProjectCreate) -> Project:
        try:
            logger.info(f"Creating project: {project_in.name}")
            
            db_project = Project(
                id=str(uuid4()),
                name=project_in.name,
                description=project_in.description,
                data={}
            )
            
            self.db.add(db_project)
            await self.db.commit()
            await self.db.refresh(db_project)
            
            logger.info(f"Project created with ID: {db_project.id}")

            # Create project directory structure
            project_dir = os.path.join("data/projects", str(db_project.id))
            os.makedirs(project_dir, exist_ok=True)
            os.makedirs(os.path.join(project_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(project_dir, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(project_dir, "exports"), exist_ok=True)
            
            # Update project with directory path
            db_project.data = {"project_dir": project_dir}
            await self.db.commit()
            await self.db.refresh(db_project)
            
            logger.info(f"Project directory created: {project_dir}")
            return db_project
            
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            await self.db.rollback()
            raise

    async def get_all_projects(self) -> list[Project]:
        try:
            result = await self.db.execute(select(Project))
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting all projects: {e}")
            return []

    async def get_project(self, project_id: str) -> Project | None:
        try:
            result = await self.db.execute(select(Project).filter(Project.id == project_id))
            return result.scalars().first()
        except Exception as e:
            logger.error(f"Error getting project {project_id}: {e}")
            return None

    async def update_project(self, project_id: str, project_in: ProjectUpdate) -> Project | None:
        try:
            project = await self.get_project(project_id)
            if not project:
                return None
            for var, value in vars(project_in).items():
                setattr(project, var, value)
            await self.db.commit()
            await self.db.refresh(project)
            return project
        except Exception as e:
            logger.error(f"Error updating project {project_id}: {e}")
            await self.db.rollback()
            return None

    async def delete_project(self, project_id: str) -> bool:
        try:
            project = await self.get_project(project_id)
            if not project:
                return False
            
            project_dir = project.data.get("project_dir")
            if project_dir and os.path.exists(project_dir):
                shutil.rmtree(project_dir, ignore_errors=True)

            await self.db.delete(project)
            await self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Error deleting project {project_id}: {e}")
            await self.db.rollback()
            return False

    async def update_project_data(self, project_id: str, new_data: dict) -> Project | None:
        try:
            project = await self.get_project(project_id)
            if not project:
                return None
            project.data = {**project.data, **new_data}
            await self.db.commit()
            await self.db.refresh(project)
            return project
        except Exception as e:
            logger.error(f"Error updating project data for {project_id}: {e}")
            await self.db.rollback()
            return None

class JobService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_job(self, job_in: TrainingJobCreate) -> TrainingJob:
        try:
            db_job = TrainingJob(
                id=str(uuid4()),
                project_id=job_in.project_id,
                status="pending",
                metrics={}
            )
            self.db.add(db_job)
            await self.db.commit()
            await self.db.refresh(db_job)
            return db_job
        except Exception as e:
            logger.error(f"Error creating job: {e}")
            await self.db.rollback()
            raise

    async def get_job(self, job_id: str) -> TrainingJob | None:
        try:
            result = await self.db.execute(select(TrainingJob).filter(TrainingJob.id == job_id))
            return result.scalars().first()
        except Exception as e:
            logger.error(f"Error getting job {job_id}: {e}")
            return None

    async def update_job_status(self, job_id: str, status: str) -> TrainingJob | None:
        try:
            job = await self.get_job(job_id)
            if not job:
                return None
            job.status = status
            await self.db.commit()
            await self.db.refresh(job)
            return job
        except Exception as e:
            logger.error(f"Error updating job status {job_id}: {e}")
            await self.db.rollback()
            return None

    async def update_job_progress(self, job_id: str, progress_data: dict) -> TrainingJob | None:
        try:
            job = await self.get_job(job_id)
            if not job:
                return None
            
            # Extract progress percentage from the progress_data dictionary
            progress_percentage = progress_data.get('progress', 0.0)
            if isinstance(progress_percentage, (int, float)):
                job.progress = float(progress_percentage)
            else:
                job.progress = 0.0
            
            # Store detailed progress data in metrics
            job.metrics = {**job.metrics, **progress_data}
            
            await self.db.commit()
            await self.db.refresh(job)
            return job
        except Exception as e:
            logger.error(f"Error updating job progress {job_id}: {e}")
            await self.db.rollback()
            return None

    async def get_jobs_for_project(self, project_id: str) -> list[TrainingJob]:
        try:
            result = await self.db.execute(select(TrainingJob).filter(TrainingJob.project_id == project_id))
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting jobs for project {project_id}: {e}")
            return []