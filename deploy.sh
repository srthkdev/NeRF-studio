#!/bin/bash

# NeRF Studio Deployment Script
# This script helps deploy the application to various platforms

set -e

echo "🚀 NeRF Studio Deployment Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_status "Docker is installed"
}

# Check if Docker Compose is installed
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_status "Docker Compose is installed"
}

# Build and run with Docker Compose
deploy_docker() {
    print_status "Building and deploying with Docker Compose..."
    
    # Stop existing containers
    docker-compose down 2>/dev/null || true
    
    # Build and start services
    docker-compose up --build -d
    
    print_status "Docker deployment completed!"
    print_status "Frontend: http://localhost"
    print_status "Backend: http://localhost:8000"
    print_status "API Docs: http://localhost:8000/docs"
}

# Deploy to Render (instructions)
deploy_render() {
    print_status "Render deployment instructions:"
    echo ""
    echo "1. Push your code to GitHub:"
    echo "   git add ."
    echo "   git commit -m 'Deploy to Render'"
    echo "   git push origin main"
    echo ""
    echo "2. Go to https://render.com"
    echo "3. Click 'New +' → 'Blueprint'"
    echo "4. Connect your GitHub repository"
    echo "5. Render will automatically detect render.yaml"
    echo "6. Click 'Apply' to deploy"
    echo ""
    print_status "Your application will be deployed to Render!"
}

# Local development setup
setup_local() {
    print_status "Setting up local development environment..."
    
    # Backend setup
    if [ ! -d "backend/venv" ]; then
        print_status "Creating Python virtual environment..."
        cd backend
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
        cd ..
    fi
    
    # Frontend setup
    if [ ! -d "frontend/node_modules" ]; then
        print_status "Installing frontend dependencies..."
        cd frontend
        npm install
        cd ..
    fi
    
    print_status "Local development environment is ready!"
    print_status "To start development:"
    echo "  Backend: cd backend && source venv/bin/activate && python -m uvicorn app.main:app --reload"
    echo "  Frontend: cd frontend && npm run dev"
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  docker    - Deploy using Docker Compose"
    echo "  render    - Show Render deployment instructions"
    echo "  local     - Setup local development environment"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 docker    # Deploy with Docker"
    echo "  $0 render    # Show Render instructions"
    echo "  $0 local     # Setup local development"
}

# Main script logic
case "${1:-help}" in
    "docker")
        check_docker
        check_docker_compose
        deploy_docker
        ;;
    "render")
        deploy_render
        ;;
    "local")
        setup_local
        ;;
    "help"|*)
        show_usage
        ;;
esac 