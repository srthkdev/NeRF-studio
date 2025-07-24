#!/bin/bash

echo "🔧 Setting up NeRF Studio Backend with SQLite..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

echo "🗄️ Initializing database..."
python3 init_db.py

echo "✅ Setup complete!"
echo ""
echo "🚀 To start the server, run:"
echo "   python3 run.py"
echo ""
echo "📊 API Documentation will be available at:"
echo "   http://localhost:8000/docs" 