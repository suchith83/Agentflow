"""
Tests for Phase 3: AgentEvaluator and reporters.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentflow.evaluation import (
    AgentEvaluator,
    CriterionConfig,
    CriterionResult,
    EvalCase,
    EvalCaseResult,
    EvalConfig,
    EvalReport,
    EvalSet,
    EvalSummary,
    EvaluationRunner,
    Invocation,
    MatchType,
    MessageContent,
    TrajectoryCollector,
)
from agentflow.evaluation.reporters.console import ConsoleReporter, Colors, print_report
from agentflow.evaluation.reporters.json import JSONReporter, JUnitXMLReporter
from agentflow.evaluation.reporters.html import HTMLReporter


# ============================================================================
# AgentEvaluator Tests
# ============================================================================


class TestAgentEvaluator:
    """Tests for the AgentEvaluator class."""

    def test_init_with_default_config(self):
        """Test evaluator initializes with default config."""
        mock_graph = MagicMock()
        evaluator = AgentEvaluator(mock_graph)

        assert evaluator.graph is mock_graph
        assert evaluator.config is not None
        assert isinstance(evaluator.criteria, list)

    def test_init_with_custom_config(self):
        """Test evaluator initializes with custom config."""
        mock_graph = MagicMock()
        config = EvalConfig(
            criteria={
                "trajectory_match": CriterionConfig(threshold=0.9),
                "response_match": CriterionConfig(threshold=0.7),
            }
        )
        evaluator = AgentEvaluator(mock_graph, config=config)

        assert evaluator.config == config
        # Should have 2 enabled criteria
        assert len(evaluator.criteria) == 2

    def test_build_criteria(self):
        """Test criteria are built from config."""
        mock_graph = MagicMock()
        config = EvalConfig(
            criteria={
                "tool_trajectory_avg_score": CriterionConfig(threshold=0.8),
                "response_match_score": CriterionConfig(threshold=0.6),
            }
        )
        evaluator = AgentEvaluator(mock_graph, config=config)

        assert len(evaluator.criteria) == 2
        criterion_names = [c.name for c in evaluator.criteria]
        assert "tool_trajectory_avg_score" in criterion_names
        assert "response_match_score" in criterion_names

    def test_create_unknown_criterion(self):
        """Test unknown criterion returns None."""
        mock_graph = MagicMock()
        config = EvalConfig(
            criteria={
                "unknown_criterion": CriterionConfig(),
            }
        )
        evaluator = AgentEvaluator(mock_graph, config=config)

        # Unknown criterion should be skipped
        assert len(evaluator.criteria) == 0

    def test_load_eval_set_file_not_found(self):
        """Test loading non-existent eval set raises error."""
        mock_graph = MagicMock()
        evaluator = AgentEvaluator(mock_graph)

        with pytest.raises(FileNotFoundError):
            evaluator._load_eval_set("/nonexistent/path.json")

    def test_load_eval_set_success(self):
        """Test loading eval set from file."""
        mock_graph = MagicMock()
        evaluator = AgentEvaluator(mock_graph)

        eval_set = EvalSet(
            eval_set_id="test_set",
            name="Test Set",
            eval_cases=[
                EvalCase.single_turn(
                    eval_id="case1",
                    user_query="Hello",
                    expected_response="Hi there",
                )
            ],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(eval_set.model_dump_json())
            temp_path = f.name

        try:
            loaded = evaluator._load_eval_set(temp_path)
            assert loaded.eval_set_id == "test_set"
            assert len(loaded.eval_cases) == 1
        finally:
            Path(temp_path).unlink()

    def test_extract_response_from_messages(self):
        """Test extracting response from graph result."""
        mock_graph = MagicMock()
        evaluator = AgentEvaluator(mock_graph)

        # Test with dict message
        result = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        response = evaluator._extract_response(result)
        assert response == "Hi there!"

    def test_extract_response_empty_result(self):
        """Test extracting response from empty result."""
        mock_graph = MagicMock()
        evaluator = AgentEvaluator(mock_graph)

        response = evaluator._extract_response({})
        assert response == ""

        response = evaluator._extract_response(None)
        assert response == ""


class TestEvaluationRunner:
    """Tests for the EvaluationRunner class."""

    def test_init(self):
        """Test runner initialization."""
        runner = EvaluationRunner()
        assert runner.default_config is not None
        assert runner.results == {}

    def test_summary_empty(self):
        """Test summary with no results."""
        runner = EvaluationRunner()
        summary = runner.summary

        assert summary["total_evaluations"] == 0


# ============================================================================
# ConsoleReporter Tests
# ============================================================================


class TestConsoleReporter:
    """Tests for the ConsoleReporter class."""

    def test_init_with_color(self):
        """Test reporter initializes with color enabled."""
        reporter = ConsoleReporter(use_color=True)
        assert reporter.use_color is True

    def test_init_without_color(self):
        """Test reporter initializes with color disabled."""
        reporter = ConsoleReporter(use_color=False)
        assert reporter.use_color is False

    def test_report_prints_output(self, capsys):
        """Test report prints to stdout."""
        reporter = ConsoleReporter(use_color=False)

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[
                EvalCaseResult.success(
                    eval_id="case1",
                    criterion_results=[
                        CriterionResult.success(
                            criterion="test",
                            score=1.0,
                            threshold=0.8,
                        )
                    ],
                )
            ],
            eval_set_name="Test Set",
        )

        reporter.report(report)
        captured = capsys.readouterr()

        assert "Test Set" in captured.out
        assert "1" in captured.out  # Total cases

    def test_verbose_mode(self, capsys):
        """Test verbose mode shows more details."""
        reporter = ConsoleReporter(use_color=False, verbose=True)

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[
                EvalCaseResult.success(
                    eval_id="case1",
                    criterion_results=[
                        CriterionResult.success(
                            criterion="trajectory_match",
                            score=0.9,
                            threshold=0.8,
                        )
                    ],
                    name="Test Case 1",
                )
            ],
        )

        reporter.report(report)
        captured = capsys.readouterr()

        assert "Test Case 1" in captured.out


class TestPrintReport:
    """Test the print_report convenience function."""

    def test_print_report(self, capsys):
        """Test print_report function."""
        report = EvalReport.create(
            eval_set_id="test",
            results=[],
        )

        print_report(report, use_color=False)
        captured = capsys.readouterr()

        assert "test" in captured.out


# ============================================================================
# JSONReporter Tests
# ============================================================================


class TestJSONReporter:
    """Tests for the JSONReporter class."""

    def test_to_dict(self):
        """Test converting report to dict."""
        reporter = JSONReporter()

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[
                EvalCaseResult.success(
                    eval_id="case1",
                    criterion_results=[],
                )
            ],
        )

        data = reporter.to_dict(report)
        assert data["eval_set_id"] == "test_set"
        assert len(data["results"]) == 1

    def test_to_json(self):
        """Test converting report to JSON string."""
        reporter = JSONReporter(indent=2)

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[],
        )

        json_str = reporter.to_json(report)
        data = json.loads(json_str)
        assert data["eval_set_id"] == "test_set"

    def test_save(self):
        """Test saving report to file."""
        reporter = JSONReporter()

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            reporter.save(report, str(path))

            assert path.exists()
            with path.open() as f:
                data = json.load(f)
            assert data["eval_set_id"] == "test_set"

    def test_exclude_details(self):
        """Test excluding details from output."""
        reporter = JSONReporter(include_details=False)

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[
                EvalCaseResult.success(
                    eval_id="case1",
                    criterion_results=[
                        CriterionResult.success(
                            criterion="test",
                            score=1.0,
                            threshold=0.8,
                            details={"key": "value"},
                        )
                    ],
                )
            ],
        )

        data = reporter.to_dict(report)
        # Details should be removed
        cr = data["results"][0]["criterion_results"][0]
        assert "details" not in cr


class TestJUnitXMLReporter:
    """Tests for the JUnitXMLReporter class."""

    def test_to_xml(self):
        """Test converting report to JUnit XML."""
        reporter = JUnitXMLReporter()

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[
                EvalCaseResult.success(
                    eval_id="case1",
                    criterion_results=[],
                    name="Test Case 1",
                )
            ],
        )

        xml = reporter.to_xml(report)
        assert "<?xml" in xml
        assert "testsuite" in xml
        assert "testcase" in xml

    def test_save(self):
        """Test saving to XML file."""
        reporter = JUnitXMLReporter()

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "junit.xml"
            reporter.save(report, str(path))

            assert path.exists()
            content = path.read_text()
            assert "testsuite" in content


# ============================================================================
# HTMLReporter Tests
# ============================================================================


class TestHTMLReporter:
    """Tests for the HTMLReporter class."""

    def test_to_html(self):
        """Test converting report to HTML."""
        reporter = HTMLReporter()

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[
                EvalCaseResult.success(
                    eval_id="case1",
                    criterion_results=[],
                    name="Test Case 1",
                )
            ],
            eval_set_name="Test Set",
        )

        html = reporter.to_html(report)
        assert "<!DOCTYPE html>" in html
        assert "Test Set" in html
        assert "Test Case 1" in html

    def test_save(self):
        """Test saving to HTML file."""
        reporter = HTMLReporter()

        report = EvalReport.create(
            eval_set_id="test_set",
            results=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            reporter.save(report, str(path))

            assert path.exists()
            content = path.read_text()
            assert "<!DOCTYPE html>" in content

    def test_render_failed_case(self):
        """Test rendering a failed case."""
        reporter = HTMLReporter()

        result = EvalCaseResult(
            eval_id="failed_case",
            name="Failed Case",
            passed=False,
            criterion_results=[
                CriterionResult(
                    criterion="test",
                    score=0.5,
                    passed=False,
                    threshold=0.8,
                )
            ],
        )

        html = reporter._render_case(result)
        assert "fail" in html
        assert "Failed Case" in html

    def test_render_error_case(self):
        """Test rendering an error case."""
        reporter = HTMLReporter()

        result = EvalCaseResult.failure(
            eval_id="error_case",
            error="Something went wrong",
            name="Error Case",
        )

        html = reporter._render_case(result)
        assert "error" in html
        assert "Something went wrong" in html
