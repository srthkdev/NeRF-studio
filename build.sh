#!/bin/bash

# NeRF Studio Frontend Build Script for Render
set -e

echo "🚀 Building NeRF Studio Frontend..."

# Navigate to frontend directory
cd frontend

# Install all dependencies (including devDependencies for TypeScript)
echo "📦 Installing dependencies..."
npm install

# Build the application
echo "🔨 Building application..."
npm run build

echo "✅ Build completed successfully!"
echo "📁 Static files generated in: frontend/dist/" 