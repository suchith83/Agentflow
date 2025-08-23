#!/bin/bash

# PyAgenity Test Runner
# This script runs the pytest test suite for the PyAgenity library

echo "Running PyAgenity Test Suite..."
echo "================================"

# Run tests with coverage
uv run pytest \
  --cov=pyagenity \
  --cov-report=html \
  --cov-report=term-missing \
  --cov-report=xml \
  --cov-fail-under=0 \
  -v

echo ""
echo "Test run complete!"
echo "Coverage reports generated:"
echo "  - HTML: htmlcov/index.html"
echo "  - XML: coverage.xml"
echo "  - Terminal: shown above"
