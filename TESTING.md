# Testing Guide for PyAgenity

This document describes how to run tests for the PyAgenity library.

## Test Setup

The PyAgenity library uses pytest for testing with the following configuration:

- **Test Directory**: `tests/` (only this directory is scanned for tests)
- **Coverage Target**: `pyagenity/` module only
- **Excluded Directories**: `normal_tests/`, `examples/`, `docs/`, and other non-library directories

## Running Tests

### Option 1: Using UV directly

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=pyagenity --cov-report=html --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_basic.py -v
```

### Option 2: Using Makefile

```bash
# Run tests without coverage
make test

# Run tests with coverage reports
make test-cov
```

### Option 3: Using the test runner script

```bash
# Run the convenience script
./run_tests.sh
```

## Coverage Reports

After running tests with coverage, the following reports are generated:

- **HTML Report**: `htmlcov/index.html` - Open in browser for detailed coverage view
- **XML Report**: `coverage.xml` - For CI/CD integration
- **Terminal Report**: Shown in console with missing line numbers

## Test Structure

The test suite is organized to mirror the `pyagenity/` module structure:

```
tests/
├── __init__.py
├── test_basic.py              # Basic functionality tests
├── graph/
│   ├── __init__.py
│   └── test_graph.py          # Graph module tests
├── state/
│   ├── __init__.py
│   └── test_state.py          # State module tests
├── utils/
│   └── __init__.py
├── checkpointer/
│   └── __init__.py
├── publisher/
│   └── __init__.py
└── exceptions/
    └── __init__.py
```

## Writing New Tests

When adding new tests:

1. Create test files in the appropriate `tests/` subdirectory
2. Name test files with the pattern `test_*.py`
3. Import modules using `from pyagenity.module import ...`
4. Use `pytest` conventions for test functions (`test_*`)
5. Add `# noqa: S101` comment after assertions to suppress bandit warnings in test files

## Configuration

The pytest configuration is in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests/"]
addopts = [
    "--cov=pyagenity",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-report=xml",
    "--cov-fail-under=0",
    "--strict-markers",
    "-v"
]

[tool.coverage.run]
source = ["pyagenity"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
    "normal_tests/*",
    "examples/*",
    "docs/*"
]
```

This ensures that:
- Only the `tests/` directory is scanned for tests
- Coverage is measured only for the `pyagenity/` module
- Excluded directories don't affect coverage percentages
