#!/usr/bin/env python3
"""
Helper script to run tests for the react_sync example.

This script provides convenient commands for running different test suites
with various options.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd: list[str]) -> int:
    """Run a command and return the exit code."""
    print(f"\n{'=' * 70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run tests for react_sync example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_tests.py all
  
  # Run only unit tests
  python run_tests.py unit
  
  # Run only evaluation tests
  python run_tests.py eval
  
  # Run with coverage report
  python run_tests.py all --coverage
  
  # Run integration tests (requires API key)
  python run_tests.py integration
  
  # Run specific test class
  python run_tests.py specific TestGetWeatherTool
  
  # Run with verbose output
  python run_tests.py all -v
        """,
    )

    parser.add_argument(
        "suite",
        choices=["all", "unit", "eval", "evaluation", "integration", "quick", "specific"],
        help="Test suite to run",
    )

    parser.add_argument(
        "test_name", nargs="?", help="Specific test class or function to run (for suite=specific)"
    )

    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")

    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    parser.add_argument("--markers", action="store_true", help="Show available test markers")

    parser.add_argument(
        "--collect-only", action="store_true", help="Only collect tests, don't run them"
    )

    args = parser.parse_args()

    # Base pytest command
    base_cmd = ["pytest"]

    # Handle markers display
    if args.markers:
        return run_command(["pytest", "--markers"])

    # Handle collect only
    if args.collect_only:
        base_cmd.append("--collect-only")

    # Add verbosity
    if args.verbose:
        base_cmd.append("-vv")

    # Build command based on suite
    if args.suite == "all":
        cmd = base_cmd + ["test_react_sync.py", "test_react_sync_evaluation.py"]

    elif args.suite == "unit":
        cmd = base_cmd + ["test_react_sync.py"]

    elif args.suite in ["eval", "evaluation"]:
        cmd = base_cmd + ["test_react_sync_evaluation.py"]

    elif args.suite == "integration":
        cmd = base_cmd + [
            "test_react_sync_evaluation.py::TestAgentResponseQuality::test_agent_uses_tool_when_appropriate",
            "--run-integration",
        ]
        print("\n⚠️  Integration tests require GOOGLE_API_KEY environment variable")
        print("Set it with: export GOOGLE_API_KEY='your-key-here'\n")

    elif args.suite == "quick":
        # Run only fast tests
        cmd = base_cmd + ["test_react_sync.py", "test_react_sync_evaluation.py", "-m", "not slow"]

    elif args.suite == "specific":
        if not args.test_name:
            print("Error: test_name is required when suite='specific'")
            print("Example: python run_tests.py specific TestGetWeatherTool")
            return 1

        # Try to find the test in either file
        cmd = base_cmd + [
            f"test_react_sync.py::{args.test_name}",
            f"test_react_sync_evaluation.py::{args.test_name}",
            "-k",
            args.test_name,
        ]

    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=react_sync", "--cov-report=term-missing"])

        if args.html:
            cmd.append("--cov-report=html")

    # Run the command
    exit_code = run_command(cmd)

    # If coverage HTML was generated, show message
    if args.coverage and args.html and exit_code == 0:
        print("\n" + "=" * 70)
        print("📊 Coverage report generated!")
        print("Open htmlcov/index.html to view the report")
        print("=" * 70)

    # Print summary
    if exit_code == 0:
        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ Some tests failed!")
        print("=" * 70)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
