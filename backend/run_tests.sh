#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set Python path to include the project root
export PYTHONPATH=/Users/sarthak/Projects/ml_projects/NeRF

# Run tests
python -m pytest tests/ -v "$@" 