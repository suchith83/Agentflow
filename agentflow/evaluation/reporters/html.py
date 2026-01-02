"""
HTML reporter for evaluation results.

Generates interactive HTML reports for evaluation results.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from agentflow.evaluation.eval_result import EvalCaseResult, EvalReport


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --color-pass: #22c55e;
            --color-fail: #ef4444;
            --color-warn: #f59e0b;
            --color-bg: #f8fafc;
            --color-card: #ffffff;
            --color-border: #e2e8f0;
            --color-text: #1e293b;
            --color-muted: #64748b;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                         Oxygen, Ubuntu, sans-serif;
            background-color: var(--color-bg);
            color: var(--color-text);
            line-height: 1.6;
            padding: 2rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            margin-bottom: 2rem;
        }}

        h1 {{
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }}

        .timestamp {{
            color: var(--color-muted);
            font-size: 0.875rem;
        }}

        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .stat-card {{
            background: var(--color-card);
            border: 1px solid var(--color-border);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }}

        .stat-value {{
            font-size: 2rem;
            font-weight: 700;
        }}

        .stat-label {{
            color: var(--color-muted);
            font-size: 0.875rem;
        }}

        .stat-pass .stat-value {{ color: var(--color-pass); }}
        .stat-fail .stat-value {{ color: var(--color-fail); }}
        .stat-warn .stat-value {{ color: var(--color-warn); }}

        .progress-bar {{
            background: var(--color-border);
            border-radius: 999px;
            height: 8px;
            margin-top: 0.5rem;
            overflow: hidden;
        }}

        .progress-fill {{
            background: var(--color-pass);
            height: 100%;
            transition: width 0.3s ease;
        }}

        section {{
            margin-bottom: 2rem;
        }}

        h2 {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
        }}

        .case-list {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .case-item {{
            background: var(--color-card);
            border: 1px solid var(--color-border);
            border-radius: 8px;
            padding: 1rem;
            cursor: pointer;
            transition: box-shadow 0.2s;
        }}

        .case-item:hover {{
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }}

        .case-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .case-status {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: white;
            font-size: 0.75rem;
        }}

        .case-status.pass {{ background: var(--color-pass); }}
        .case-status.fail {{ background: var(--color-fail); }}
        .case-status.error {{ background: var(--color-warn); }}

        .case-name {{
            flex: 1;
            font-weight: 500;
        }}

        .case-duration {{
            color: var(--color-muted);
            font-size: 0.875rem;
        }}

        .case-details {{
            display: none;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid var(--color-border);
        }}

        .case-item.expanded .case-details {{
            display: block;
        }}

        .criterion-list {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .criterion-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem;
            background: var(--color-bg);
            border-radius: 4px;
        }}

        .criterion-icon {{
            font-size: 0.875rem;
        }}

        .criterion-name {{
            flex: 1;
        }}

        .criterion-score {{
            font-family: monospace;
        }}

        .error-message {{
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-radius: 4px;
            padding: 0.75rem;
            color: #991b1b;
            margin-top: 0.5rem;
            font-family: monospace;
            font-size: 0.875rem;
        }}

        .filter-bar {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}

        .filter-btn {{
            padding: 0.5rem 1rem;
            border: 1px solid var(--color-border);
            background: var(--color-card);
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.875rem;
        }}

        .filter-btn.active {{
            background: var(--color-text);
            color: white;
        }}

        footer {{
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--color-border);
            color: var(--color-muted);
            font-size: 0.875rem;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 {title}</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>

        <section class="summary">
            <div class="stat-card">
                <div class="stat-value">{total_cases}</div>
                <div class="stat-label">Total Cases</div>
            </div>
            <div class="stat-card stat-pass">
                <div class="stat-value">{passed_cases}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-card stat-fail">
                <div class="stat-value">{failed_cases}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card stat-warn">
                <div class="stat-value">{error_cases}</div>
                <div class="stat-label">Errors</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{pass_rate_pct}%</div>
                <div class="stat-label">Pass Rate</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {pass_rate_pct}%"></div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{duration}s</div>
                <div class="stat-label">Duration</div>
            </div>
        </section>

        <section>
            <h2>Test Cases</h2>
            <div class="filter-bar">
                <button class="filter-btn active" data-filter="all">All</button>
                <button class="filter-btn" data-filter="pass">Passed</button>
                <button class="filter-btn" data-filter="fail">Failed</button>
                <button class="filter-btn" data-filter="error">Errors</button>
            </div>
            <div class="case-list">
{case_items}
            </div>
        </section>

        <footer>
            <p>Generated by Agentflow Evaluation Framework</p>
        </footer>
    </div>

    <script>
        // Toggle case details
        document.querySelectorAll('.case-item').forEach(item => {{
            item.addEventListener('click', () => {{
                item.classList.toggle('expanded');
            }});
        }});

        // Filter buttons
        document.querySelectorAll('.filter-btn').forEach(btn => {{
            btn.addEventListener('click', () => {{
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                const filter = btn.dataset.filter;
                document.querySelectorAll('.case-item').forEach(item => {{
                    if (filter === 'all') {{
                        item.style.display = '';
                    }} else {{
                        item.style.display = item.dataset.status === filter ? '' : 'none';
                    }}
                }});
            }});
        }});
    </script>
</body>
</html>
"""


CASE_ITEM_TEMPLATE = """
            <div class="case-item" data-status="{status}">
                <div class="case-header">
                    <div class="case-status {status}">{icon}</div>
                    <span class="case-name">{name}</span>
                    <span class="case-duration">{duration}s</span>
                </div>
                <div class="case-details">
                    <div class="criterion-list">
{criteria_items}
                    </div>
{error_section}
                </div>
            </div>
"""


class HTMLReporter:
    """Generate interactive HTML reports for evaluation results.

    Creates a self-contained HTML file with styled output and
    interactive features for exploring test results.

    Example:
        ```python
        reporter = HTMLReporter()
        reporter.save(report, "results/report.html")
        ```
    """

    def __init__(self, include_details: bool = True):
        """Initialize the HTML reporter.

        Args:
            include_details: Whether to include criterion details.
        """
        self.include_details = include_details

    def to_html(self, report: EvalReport) -> str:
        """Convert report to HTML string.

        Args:
            report: The evaluation report.

        Returns:
            Complete HTML document as string.
        """
        title = report.eval_set_name or report.eval_set_id
        timestamp = datetime.fromtimestamp(report.timestamp).strftime("%Y-%m-%d %H:%M:%S")

        # Generate case items
        case_items = []
        for result in report.results:
            case_items.append(self._render_case(result))

        return HTML_TEMPLATE.format(
            title=html.escape(title),
            timestamp=timestamp,
            total_cases=report.summary.total_cases,
            passed_cases=report.summary.passed_cases,
            failed_cases=report.summary.failed_cases,
            error_cases=report.summary.error_cases,
            pass_rate_pct=f"{report.summary.pass_rate * 100:.0f}",
            duration=f"{report.summary.total_duration_seconds:.2f}",
            case_items="\n".join(case_items),
        )

    def _render_case(self, result: EvalCaseResult) -> str:
        """Render a single case item."""
        if result.is_error:
            status = "error"
            icon = "!"
        elif result.passed:
            status = "pass"
            icon = "✓"
        else:
            status = "fail"
            icon = "✗"

        # Render criteria
        criteria_items = []
        for cr in result.criterion_results:
            cr_icon = "✓" if cr.passed else "✗"
            criteria_items.append(
                f'                        <div class="criterion-item">'
                f'<span class="criterion-icon">{cr_icon}</span>'
                f'<span class="criterion-name">{html.escape(cr.criterion)}</span>'
                f'<span class="criterion-score">{cr.score:.2f} / {cr.threshold}</span>'
                f"</div>"
            )

        # Error section
        error_section = ""
        if result.error:
            error_section = (
                f'                    <div class="error-message">{html.escape(result.error)}</div>'
            )

        return CASE_ITEM_TEMPLATE.format(
            status=status,
            icon=icon,
            name=html.escape(result.name or result.eval_id),
            duration=f"{result.duration_seconds:.2f}",
            criteria_items="\n".join(criteria_items),
            error_section=error_section,
        )

    def save(self, report: EvalReport, path: str) -> None:
        """Save report to HTML file.

        Args:
            report: The evaluation report.
            path: Output file path.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8") as f:
            f.write(self.to_html(report))

    @classmethod
    def quick_save(cls, report: EvalReport, path: str) -> None:
        """Convenience method to quickly save a report.

        Args:
            report: The evaluation report.
            path: Output file path.
        """
        reporter = cls()
        reporter.save(report, path)
