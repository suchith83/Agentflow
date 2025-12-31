"""
JSON reporter for evaluation results.

Exports evaluation reports to JSON files with optional formatting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentflow.evaluation.eval_result import EvalReport


class JSONReporter:
    """Export evaluation results to JSON files.

    Provides options for formatting, filtering, and customizing
    the JSON output structure.

    Attributes:
        indent: JSON indentation level (None for compact).
        include_details: Whether to include full criterion details.
        include_trajectory: Whether to include trajectory data.

    Example:
        ```python
        reporter = JSONReporter(indent=2)
        reporter.save(report, "results/eval_report.json")

        # Or get as string
        json_str = reporter.to_json(report)
        ```
    """

    def __init__(
        self,
        indent: int | None = 2,
        include_details: bool = True,
        include_trajectory: bool = True,
    ):
        """Initialize the JSON reporter.

        Args:
            indent: JSON indentation (None for minified output).
            include_details: Include criterion details in output.
            include_trajectory: Include trajectory data in output.
        """
        self.indent = indent
        self.include_details = include_details
        self.include_trajectory = include_trajectory

    def to_dict(self, report: EvalReport) -> dict[str, Any]:
        """Convert report to dictionary.

        Args:
            report: The evaluation report.

        Returns:
            Dictionary representation of the report.
        """
        data = report.model_dump()

        # Optionally filter out details
        if not self.include_details:
            for result in data.get("results", []):
                for cr in result.get("criterion_results", []):
                    cr.pop("details", None)

        # Optionally filter out trajectory
        if not self.include_trajectory:
            for result in data.get("results", []):
                result.pop("actual_trajectory", None)

        return data

    def to_json(self, report: EvalReport) -> str:
        """Convert report to JSON string.

        Args:
            report: The evaluation report.

        Returns:
            JSON string representation.
        """
        data = self.to_dict(report)
        return json.dumps(data, indent=self.indent, default=str)

    def save(self, report: EvalReport, path: str) -> None:
        """Save report to a JSON file.

        Args:
            report: The evaluation report.
            path: Output file path.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8") as f:
            f.write(self.to_json(report))

    @classmethod
    def quick_save(
        cls,
        report: EvalReport,
        path: str,
        indent: int = 2,
    ) -> None:
        """Convenience method to quickly save a report.

        Args:
            report: The evaluation report.
            path: Output file path.
            indent: JSON indentation.
        """
        reporter = cls(indent=indent)
        reporter.save(report, path)


class JUnitXMLReporter:
    """Export evaluation results to JUnit XML format.

    Useful for CI/CD integration with tools that understand
    JUnit test result format.

    Example:
        ```python
        reporter = JUnitXMLReporter()
        reporter.save(report, "results/junit.xml")
        ```
    """

    def __init__(self, suite_name: str = "agent-evaluation"):
        """Initialize the JUnit reporter.

        Args:
            suite_name: Name of the test suite.
        """
        self.suite_name = suite_name

    def to_xml(self, report: EvalReport) -> str:
        """Convert report to JUnit XML format.

        Args:
            report: The evaluation report.

        Returns:
            XML string in JUnit format.
        """
        import xml.etree.ElementTree as ET
        from datetime import datetime

        # Create root element
        testsuite = ET.Element("testsuite")
        testsuite.set("name", self.suite_name)
        testsuite.set("tests", str(report.summary.total_cases))
        testsuite.set("failures", str(report.summary.failed_cases))
        testsuite.set("errors", str(report.summary.error_cases))
        testsuite.set("time", f"{report.summary.total_duration_seconds:.3f}")
        testsuite.set("timestamp", datetime.fromtimestamp(report.timestamp).isoformat())

        # Add test cases
        for result in report.results:
            testcase = ET.SubElement(testsuite, "testcase")
            testcase.set("name", result.name or result.eval_id)
            testcase.set("classname", report.eval_set_id)
            testcase.set("time", f"{result.duration_seconds:.3f}")

            if result.is_error:
                error = ET.SubElement(testcase, "error")
                error.set("message", result.error or "Unknown error")
                error.text = result.error
            elif not result.passed:
                # Add failure for each failed criterion
                for cr in result.failed_criteria:
                    failure = ET.SubElement(testcase, "failure")
                    failure.set("type", cr.criterion)
                    failure.set("message", f"Score {cr.score:.2f} below threshold {cr.threshold}")
                    if cr.error:
                        failure.text = cr.error

        # Convert to string
        return ET.tostring(testsuite, encoding="unicode", xml_declaration=True)

    def save(self, report: EvalReport, path: str) -> None:
        """Save report to JUnit XML file.

        Args:
            report: The evaluation report.
            path: Output file path.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8") as f:
            f.write(self.to_xml(report))
