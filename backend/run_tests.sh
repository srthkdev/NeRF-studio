#!/bin/bash

# NeRF Studio Test Runner
# Runs all tests and provides coverage reporting

set -e

echo "🧪 NeRF Studio Test Suite"
echo "=========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "backend/requirements.txt" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "backend/venv" ]; then
    print_status "Creating virtual environment..."
    cd backend
    python3 -m venv venv
    cd ..
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source backend/venv/bin/activate

# Install dependencies
print_status "Installing dependencies..."
cd backend
pip install -r requirements.txt
pip install pytest-cov pytest-mock httpx

# Run tests with coverage
print_status "Running tests with coverage..."
python -m pytest tests/ \
    --cov=app \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-fail-under=70 \
    -v \
    --tb=short

# Check if tests passed
if [ $? -eq 0 ]; then
    print_success "All tests passed!"
else
    print_error "Some tests failed!"
    exit 1
fi

# Run specific test categories
echo ""
print_status "Running specific test categories..."

# Unit tests
print_status "Running unit tests..."
python -m pytest tests/test_validation.py tests/test_nerf_model.py tests/test_volume_rendering.py tests/test_advanced_export.py tests/test_performance_monitor.py -v

# Integration tests
print_status "Running integration tests..."
python -m pytest tests/test_integration.py -v

# API tests
print_status "Running API tests..."
python -m pytest tests/test_api.py -v

# Performance tests (if available)
if [ -f "tests/test_performance.py" ]; then
    print_status "Running performance tests..."
    python -m pytest tests/test_performance.py -v
fi

# Advanced export tests
print_status "Running advanced export tests..."
python -m pytest tests/test_advanced_export.py -v

# Performance monitoring tests
print_status "Running performance monitoring tests..."
python -m pytest tests/test_performance_monitor.py -v

# Generate test report
echo ""
print_status "Generating test report..."
python -m pytest tests/ \
    --html=test_report.html \
    --self-contained-html \
    --cov=app \
    --cov-report=html:coverage_html \
    --cov-report=term-missing

# Show coverage summary
echo ""
print_status "Coverage Summary:"
python -m coverage report

# Check for any warnings or issues
echo ""
print_status "Checking for code quality issues..."

# Run flake8 if available
if command -v flake8 &> /dev/null; then
    print_status "Running flake8..."
    flake8 app/ --max-line-length=100 --ignore=E501,W503
else
    print_warning "flake8 not found. Install it for code quality checks."
fi

# Run mypy if available
if command -v mypy &> /dev/null; then
    print_status "Running mypy..."
    mypy app/ --ignore-missing-imports
else
    print_warning "mypy not found. Install it for type checking."
fi

# Summary
echo ""
echo "=========================="
print_success "Test suite completed!"
echo ""
print_status "Test results available in:"
echo "  - test_report.html (detailed test report)"
echo "  - coverage_html/ (coverage report)"
echo "  - .coverage (coverage data)"
echo ""
print_status "To view coverage report:"
echo "  open coverage_html/index.html"
echo ""
print_status "To view test report:"
echo "  open test_report.html"

# Deactivate virtual environment
deactivate

print_success "Test runner finished successfully!" 