#!/bin/bash

# NeRF Studio Frontend Deployment Script
# This script builds and deploys the frontend static site

set -e  # Exit on any error

echo "🚀 NeRF Studio Frontend Deployment"
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

# Check if we're in the right directory
if [ ! -f "frontend/package.json" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node --version)
print_status "Node.js version: $NODE_VERSION"

# Check npm version
NPM_VERSION=$(npm --version)
print_status "npm version: $NPM_VERSION"

# Navigate to frontend directory
cd frontend

print_status "Installing dependencies..."
npm ci

print_status "Building frontend..."
npm run build

# Check if build was successful
if [ ! -d "dist" ]; then
    print_error "Build failed - dist directory not found"
    exit 1
fi

print_status "Build completed successfully!"
print_status "Static files generated in: frontend/dist/"

# Show build statistics
echo ""
print_status "Build Statistics:"
echo "==================="
echo "Total files in dist: $(find dist -type f | wc -l)"
echo "Total size: $(du -sh dist | cut -f1)"
echo "Main bundle: $(ls -lh dist/assets/*.js | head -1 | awk '{print $5}')"

# Check for environment variables
echo ""
print_status "Environment Variables Check:"
echo "=================================="

if [ -n "$VITE_API_URL" ]; then
    print_status "VITE_API_URL: $VITE_API_URL"
else
    print_warning "VITE_API_URL not set - using default"
fi

if [ -n "$VITE_ENVIRONMENT" ]; then
    print_status "VITE_ENVIRONMENT: $VITE_ENVIRONMENT"
else
    print_warning "VITE_ENVIRONMENT not set - using development"
fi

# Preview instructions
echo ""
print_status "Next Steps:"
echo "============="
echo "1. Test locally: npm run preview"
echo "2. Deploy to Render: git push origin main"
echo "3. Or deploy manually to your hosting provider"

# Optional: Start preview server
read -p "Would you like to start the preview server? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Starting preview server on http://localhost:4173"
    npm run preview
fi

print_status "Deployment script completed successfully! 🎉" 