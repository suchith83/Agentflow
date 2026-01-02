"""
Console reporter for evaluation results.

Provides pretty-printed console output for evaluation reports.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TextIO


if TYPE_CHECKING:
    from agentflow.evaluation.eval_result import EvalCaseResult, EvalReport


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"

    @classmethod
    def disable(cls) -> None:
        """Disable all colors."""
        cls.RESET = ""
        cls.BOLD = ""
        cls.DIM = ""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.MAGENTA = ""
        cls.CYAN = ""
        cls.WHITE = ""
        cls.BG_RED = ""
        cls.BG_GREEN = ""


class ConsoleReporter:
    """Pretty-print evaluation results to console.

    Provides colored, formatted output for evaluation reports
    with configurable verbosity levels.

    Attributes:
        use_color: Whether to use ANSI colors.
        verbose: Whether to show detailed output.
        output: Output stream (default: stdout).

    Example:
        ```python
        reporter = ConsoleReporter(verbose=True)
        reporter.report(eval_report)
        ```
    """

    def __init__(
        self,
        use_color: bool = True,
        verbose: bool = False,
        output: TextIO | None = None,
    ):
        """Initialize the console reporter.

        Args:
            use_color: Whether to use ANSI colors in output.
            verbose: Whether to show detailed output.
            output: Output stream (default: stdout).
        """
        self.use_color = use_color
        self.verbose = verbose
        self.output = output or sys.stdout

        if not use_color:
            Colors.disable()

    def report(self, report: EvalReport) -> None:
        """Print a complete evaluation report.

        Args:
            report: The evaluation report to display.
        """
        self._print_header(report)
        self._print_summary(report)
        self._print_criterion_stats(report)

        if self.verbose or report.failed_cases:
            self._print_case_details(report)

        self._print_footer(report)

    def _print(self, *args, **kwargs) -> None:
        """Print to configured output stream."""
        print(*args, file=self.output, **kwargs)

    def _print_header(self, report: EvalReport) -> None:
        """Print report header."""
        title = report.eval_set_name or report.eval_set_id
        self._print()
        self._print(f"{Colors.BOLD}{Colors.CYAN}╔{'═' * 60}╗{Colors.RESET}")
        title_str = f"{Colors.BOLD}Evaluation Report: {title}{Colors.RESET}"
        self._print(f"{Colors.BOLD}{Colors.CYAN}║{Colors.RESET} {title_str}")
        self._print(f"{Colors.BOLD}{Colors.CYAN}╚{'═' * 60}╝{Colors.RESET}")
        self._print()

    def _print_summary(self, report: EvalReport) -> None:
        """Print summary statistics."""
        summary = report.summary

        # Overall status
        if summary.pass_rate == 1.0:
            status = f"{Colors.BG_GREEN}{Colors.WHITE}{Colors.BOLD} ALL PASSED {Colors.RESET}"
        elif summary.pass_rate == 0.0:
            status = f"{Colors.BG_RED}{Colors.WHITE}{Colors.BOLD} ALL FAILED {Colors.RESET}"
        else:
            status = f"{Colors.YELLOW}{Colors.BOLD}PARTIAL{Colors.RESET}"

        self._print(f"{Colors.BOLD}Summary:{Colors.RESET} {status}")
        self._print()

        # Stats table
        total_str = f"Total Cases:  {Colors.BOLD}{summary.total_cases}{Colors.RESET}"
        self._print(f"  {Colors.DIM}├─{Colors.RESET} {total_str}")

        pass_color = Colors.GREEN if summary.passed_cases > 0 else Colors.DIM
        pass_str = f"{pass_color}{summary.passed_cases} ({summary.pass_rate:.1%}){Colors.RESET}"
        self._print(f"  {Colors.DIM}├─{Colors.RESET} Passed:       {pass_str}")

        fail_color = Colors.RED if summary.failed_cases > 0 else Colors.DIM
        fail_str = f"{fail_color}{summary.failed_cases}{Colors.RESET}"
        self._print(f"  {Colors.DIM}├─{Colors.RESET} Failed:       {fail_str}")

        error_color = Colors.YELLOW if summary.error_cases > 0 else Colors.DIM
        error_str = f"{error_color}{summary.error_cases}{Colors.RESET}"
        self._print(f"  {Colors.DIM}├─{Colors.RESET} Errors:       {error_str}")

        duration_str = (
            f"{summary.total_duration_seconds:.2f}s (avg: {summary.avg_duration_seconds:.2f}s)"
        )
        self._print(f"  {Colors.DIM}└─{Colors.RESET} Duration:     {duration_str}")
        self._print()

    def _print_criterion_stats(self, report: EvalReport) -> None:
        """Print per-criterion statistics."""
        if not report.summary.criterion_stats:
            return

        self._print(f"{Colors.BOLD}Criteria Results:{Colors.RESET}")
        self._print()

        for criterion, stats in report.summary.criterion_stats.items():
            pass_rate = stats.get("pass_rate", 0.0)
            avg_score = stats.get("avg_score", 0.0)
            passed = stats.get("passed", 0)
            total = stats.get("total", 0)

            # Color based on pass rate
            HIGH_PASS_RATE = 0.9
            MED_PASS_RATE = 0.5
            if pass_rate >= HIGH_PASS_RATE:
                color = Colors.GREEN
                icon = "✓"
            elif pass_rate >= MED_PASS_RATE:
                color = Colors.YELLOW
                icon = "○"
            else:
                color = Colors.RED
                icon = "✗"

            self._print(
                f"  {color}{icon}{Colors.RESET} {criterion}: "
                f"{passed}/{total} passed, avg score: {avg_score:.2f}"
            )

        self._print()

    def _print_case_details(self, report: EvalReport) -> None:
        """Print detailed case results."""
        self._print(f"{Colors.BOLD}Case Details:{Colors.RESET}")
        self._print()

        for result in report.results:
            self._print_case(result)

    def _print_case(self, result: EvalCaseResult) -> None:
        """Print a single case result."""
        # Status icon and color
        if result.is_error:
            icon = "⚠"
            color = Colors.YELLOW
            status = "ERROR"
        elif result.passed:
            icon = "✓"
            color = Colors.GREEN
            status = "PASS"
        else:
            icon = "✗"
            color = Colors.RED
            status = "FAIL"

        name = result.name or result.eval_id
        self._print(
            f"  {color}{icon} {status}{Colors.RESET} {name} ({result.duration_seconds:.2f}s)"
        )

        # Print error if present
        if result.error:
            self._print(f"      {Colors.YELLOW}Error: {result.error}{Colors.RESET}")

        # Print failed criteria if verbose or failed
        if not result.passed or self.verbose:
            for cr in result.criterion_results:
                if not cr.passed or self.verbose:
                    cr_icon = "✓" if cr.passed else "✗"
                    cr_color = Colors.GREEN if cr.passed else Colors.RED
                    self._print(
                        f"      {cr_color}{cr_icon}{Colors.RESET} {cr.criterion}: "
                        f"{cr.score:.2f} (threshold: {cr.threshold})"
                    )
                    if cr.error:
                        self._print(f"        {Colors.YELLOW}Error: {cr.error}{Colors.RESET}")

        self._print()

    def _print_footer(self, report: EvalReport) -> None:
        """Print report footer."""
        from datetime import datetime

        timestamp = datetime.fromtimestamp(report.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        self._print(f"{Colors.DIM}Report generated: {timestamp}{Colors.RESET}")
        self._print()


def print_report(report: EvalReport, verbose: bool = False, use_color: bool = True) -> None:
    """Convenience function to print a report to console.

    Args:
        report: The evaluation report to print.
        verbose: Whether to show detailed output.
        use_color: Whether to use ANSI colors.
    """
    reporter = ConsoleReporter(use_color=use_color, verbose=verbose)
    reporter.report(report)
