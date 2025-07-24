#!/usr/bin/env python3
"""
NeRF Studio Project Setup Script
This script sets up the project for development and production use.
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {command}")
            print(f"Error output: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running command {command}: {e}")
        return False

def setup_backend():
    """Set up the backend environment."""
    print("Setting up backend...")
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists("backend/venv"):
        print("Creating virtual environment...")
        if not run_command("python -m venv venv", cwd="backend"):
            return False
    
    # Install requirements
    print("Installing Python dependencies...")
    pip_cmd = "backend/venv/bin/pip" if os.name != 'nt' else "backend\\venv\\Scripts\\pip"
    if not run_command(f"{pip_cmd} install -r requirements.txt", cwd="backend"):
        return False
    
    # Create database tables
    print("Creating database tables...")
    python_cmd = "backend/venv/bin/python" if os.name != 'nt' else "backend\\venv\\Scripts\\python"
    if not run_command(f"{python_cmd} create_tables.py", cwd="backend"):
        print("Warning: Database table creation failed. Make sure your DATABASE_URL is correct.")
    
    # Create necessary directories
    print("Creating project directories...")
    os.makedirs("backend/data/projects", exist_ok=True)
    os.makedirs("backend/logs", exist_ok=True)
    
    return True

def setup_frontend():
    """Set up the frontend environment."""
    print("Setting up frontend...")
    
    # Install npm dependencies
    print("Installing Node.js dependencies...")
    if not run_command("npm install", cwd="frontend"):
        return False
    
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up NeRF Studio...")
    
    # Check if we're in the right directory
    if not os.path.exists("backend") or not os.path.exists("frontend"):
        print("Error: Please run this script from the project root directory.")
        sys.exit(1)
    
    # Setup backend
    if not setup_backend():
        print("❌ Backend setup failed!")
        sys.exit(1)
    
    # Setup frontend
    if not setup_frontend():
        print("❌ Frontend setup failed!")
        sys.exit(1)
    
    print("✅ NeRF Studio setup complete!")
    print("\nTo start the application:")
    print("1. Backend: cd backend && venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    print("2. Frontend: cd frontend && npm run dev")
    print("\nThen open http://localhost:5173 in your browser.")

if __name__ == "__main__":
    main()