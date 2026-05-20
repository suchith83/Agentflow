"""HTML rendering helpers for eval case results.

Provides ``render_case`` (and supporting functions) extracted from
``HTMLReporter`` so that ``html.py`` stays thin.
"""

from __future__ import annotations

import html as _html
import json as _json
from typing import TYPE_CHECKING, Any

from agentflow.qa.evaluation.reporters._utils import (
    case_display_name,
    case_status_info,
    format_tool_calls,
)

if TYPE_CHECKING:
    from agentflow.qa.evaluation.eval_result import EvalCaseResult


# ── Score bar colour (hue: 0 = red  →  120 = green) ─────────────────────────


def _score_color(score: float) -> str:
    hue = round(max(0.0, min(1.0, score)) * 120)
    return f"hsl({hue},65%,45%)"


# ── Criterion row ─────────────────────────────────────────────────────────────


def render_criterion(cr: Any) -> str:
    """Render a single criterion result as a score-bar row."""
    passed = getattr(cr, "passed", False)
    score = float(getattr(cr, "score", 0.0))
    threshold = getattr(cr, "threshold", 1.0)
    criterion = getattr(cr, "criterion", "")
    reason = getattr(cr, "reason", None)
    details: dict[str, Any] = getattr(cr, "details", {}) or {}
    error = getattr(cr, "error", None)

    icon = "&#x2713;" if passed else "&#x2717;"
    bar_width = f"{score * 100:.1f}%"
    bar_color = _score_color(score)

    extra: list[str] = []

    if reason:
        extra.append(f'<div class="criterion-reason">{_html.escape(reason)}</div>')
    if error:
        extra.append(
            f'<div class="criterion-reason" style="color:#991b1b;">'
            f"Error: {_html.escape(error)}</div>"
        )
    if details:
        detail_items = [
            f'<span style="color:var(--color-muted);font-size:0.75rem;">'
            f"<strong>{_html.escape(str(dk))}</strong>: {_html.escape(str(dv))}</span>"
            for dk, dv in details.items()
            if dk != "reason"
        ]
        if detail_items:
            extra.append(
                '<div style="grid-column:2/-1;margin-left:0.25rem;">'
                + "<br/>".join(detail_items)
                + "</div>"
            )

    return (
        f'<div class="criterion-row"'
        f' data-criterion="{_html.escape(criterion)}"'
        f' data-score="{score:.4f}">'
        f'<span class="criterion-icon">{icon}</span>'
        f'<span class="criterion-name">{_html.escape(criterion)}</span>'
        f'<div class="score-bar-wrap">'
        f'<div class="score-bar-fill" style="width:{bar_width};background:{bar_color};"></div>'
        f"</div>"
        f'<span class="score-value">{score:.2f}/{threshold}</span>'
        f"</div>" + "".join(extra)
    )


# ── Case item template ────────────────────────────────────────────────────────

_CASE_TEMPLATE = (
    '            <div class="case-item" data-status="{status}" data-score="{avg_score:.4f}">\n'
    '                <div class="case-header">\n'
    '                    <span class="expand-icon">&#x25BA;</span>\n'
    '                    <div class="case-status {status}">{icon}</div>\n'
    '                    <span class="case-name">{name}</span>\n'
    '                    <span class="case-score">Score: {avg_score:.2f}</span>\n'
    '                    <span class="case-duration">{duration}s</span>\n'
    "                </div>\n"
    '                <div class="case-details">\n'
    "{body}"
    "                </div>\n"
    "            </div>"
)


# ── Public entry point ────────────────────────────────────────────────────────


def render_case(
    result: EvalCaseResult,
    *,
    include_details: bool = True,
    include_actual_response: bool = True,
    include_tool_call_details: bool = True,
    include_node_responses: bool = True,
    include_trajectory: bool = True,
) -> str:
    """Return a ``<div class="case-item">`` HTML block for *result*."""
    status, icon = case_status_info(result)
    name = _html.escape(case_display_name(result))
    duration = f"{result.duration_seconds:.2f}"

    crs = getattr(result, "criterion_results", []) or []
    avg_score = (sum(cr.score for cr in crs) / len(crs)) if crs else 0.0

    body_parts: list[str] = []

    # Metadata
    if getattr(result, "metadata", None):
        meta_items = [
            f'<span style="color:var(--color-muted);font-size:0.8rem;">'
            f"<strong>{_html.escape(str(mk))}</strong>: {_html.escape(str(mv))}</span>"
            for mk, mv in result.metadata.items()
        ]
        if meta_items:
            body_parts.append(
                _detail(
                    "Metadata",
                    '<div style="padding:0.5rem;">'
                    + "<br/>".join(meta_items)
                    + "</div>",
                )
            )

    # Agent response
    if include_actual_response and getattr(result, "actual_response", None):
        escaped = _html.escape(result.actual_response)
        body_parts.append(
            _detail(
                "Agent Response",
                f'<div class="response-box">{escaped}</div>',
                open_=True,
            )
        )

    # Tool calls
    if include_tool_call_details and getattr(result, "actual_tool_calls", None):
        tools = format_tool_calls(result.actual_tool_calls)
        if tools:
            rows = [
                "<tr>"
                f"<td>{_html.escape(t['name'])}</td>"
                f"<td>{_html.escape(t.get('call_id', ''))}</td>"
                f"<td>{_html.escape(t['args'])}</td>"
                f"<td>{_html.escape(t['result'])}</td>"
                "</tr>"
                for t in tools
            ]
            body_parts.append(
                _detail(
                    f"Tool Calls ({len(tools)})",
                    '<table class="tool-table"><thead><tr>'
                    "<th>Tool</th><th>Call ID</th><th>Arguments</th><th>Result</th>"
                    "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>",
                )
            )

    # Trajectory
    if include_trajectory and getattr(result, "actual_trajectory", None):
        steps = _render_trajectory_steps(result.actual_trajectory)
        if steps:
            body_parts.append(
                _detail(
                    f"Execution Trajectory ({len(steps)} steps)",
                    '<div class="trajectory-timeline">' + "\n".join(steps) + "</div>",
                )
            )

    # Node visits
    if getattr(result, "node_visits", None):
        nv_html = " &#x2192; ".join(_html.escape(nv) for nv in result.node_visits)
        body_parts.append(
            _detail(
                f"Node Visits ({len(result.node_visits)})",
                f'<div style="padding:0.5rem;font-family:monospace;font-size:0.8rem;">{nv_html}</div>',
            )
        )

    # Node responses
    if include_node_responses and getattr(result, "node_responses", None):
        boxes = _render_node_responses(result.node_responses)
        if boxes:
            body_parts.append(
                _detail(f"Node Responses ({len(boxes)})", "\n".join(boxes))
            )

    # Messages
    if getattr(result, "messages", None):
        msg_items = _render_messages(result.messages)
        if msg_items:
            body_parts.append(
                _detail(
                    f"Messages ({len(result.messages)})",
                    '<div style="padding:0.5rem;">' + "\n".join(msg_items) + "</div>",
                )
            )

    # Turn results
    if getattr(result, "turn_results", None):
        turn_items = _render_turn_results(result.turn_results)
        if turn_items:
            body_parts.append(
                _detail(
                    f"Turn-by-Turn Results ({len(result.turn_results)} turns)",
                    "\n".join(turn_items),
                )
            )

    # Criteria with score bars
    if include_details and crs:
        criteria_html = "\n".join(
            f"                        {render_criterion(cr)}" for cr in crs
        )
        body_parts.append(
            _detail(
                "Criteria",
                f'<div class="criterion-list">\n{criteria_html}\n                    </div>',
                open_=True,
            )
        )

    # Error
    if result.error:
        body_parts.append(
            f'                    <div class="error-message">'
            f"{_html.escape(result.error)}</div>"
        )

    return _CASE_TEMPLATE.format(
        status=status,
        icon=icon,
        name=name,
        avg_score=avg_score,
        duration=duration,
        body="\n".join(body_parts) + ("\n" if body_parts else ""),
    )


# ── Detail section helper ─────────────────────────────────────────────────────


def _detail(summary: str, content: str, *, open_: bool = False) -> str:
    open_attr = " open" if open_ else ""
    return (
        f'                    <details class="detail-section"{open_attr}>'
        f"<summary>{summary}</summary>"
        f"{content}"
        f"</details>"
    )


# ── Trajectory ────────────────────────────────────────────────────────────────


def _render_trajectory_steps(trajectory: list[Any]) -> list[str]:
    steps: list[str] = []
    _muted = "color:var(--color-muted);font-size:0.75rem;"

    for step in trajectory:
        if isinstance(step, dict):
            stype = step.get("step_type", step.get("type", "node")).lower()
            sname = _html.escape(
                str(step.get("name", step.get("node", step.get("tool", ""))))
            )
            sargs = step.get("args", {})
            smeta = step.get("metadata", {})
            stimestamp = step.get("timestamp")
        elif hasattr(step, "step_type"):
            stype = (
                step.step_type.value
                if hasattr(step.step_type, "value")
                else str(step.step_type)
            )
            sname = _html.escape(str(step.name))
            sargs = step.args if hasattr(step, "args") else {}
            smeta = step.metadata if hasattr(step, "metadata") else {}
            stimestamp = step.timestamp if hasattr(step, "timestamp") else None
        else:
            stype, sname, sargs, smeta, stimestamp = (
                "node",
                _html.escape(str(step)),
                {},
                {},
                None,
            )

        css_cls = "tool" if stype == "tool" else "node"
        detail_parts = [f'<span class="traj-detail">{sname}</span>']

        if sargs:
            try:
                args_str = _json.dumps(sargs, default=str, ensure_ascii=False)
            except (TypeError, ValueError):
                args_str = str(sargs)
            detail_parts.append(
                f'<span class="traj-detail" style="{_muted}"> args: {_html.escape(args_str)}</span>'
            )
        if smeta:
            detail_parts.append(
                f'<span class="traj-detail" style="{_muted}"> meta: {_html.escape(str(smeta))}</span>'
            )
        if stimestamp:
            detail_parts.append(
                f'<span class="traj-detail" style="{_muted}"> @{stimestamp}</span>'
            )

        steps.append(
            f'<div class="traj-step {css_cls}">'
            f'<span class="traj-label">[{stype.upper()}]</span>'
            + "".join(detail_parts)
            + "</div>"
        )
    return steps


# ── Node responses ────────────────────────────────────────────────────────────


def _render_node_responses(node_responses: list[Any]) -> list[str]:
    boxes: list[str] = []
    _ms = "color:var(--color-muted);font-size:0.75rem;"

    for nr in node_responses:
        if isinstance(nr, dict):
            nname = _html.escape(str(nr.get("node_name", "?")))
            nout = _html.escape(str(nr.get("response_text", "")))
            nr_tools: list[str] = nr.get("tool_call_names", [])
            nr_final: bool = nr.get("is_final", False)
            nr_has_tools: bool = nr.get("has_tool_calls", False)
            nr_timestamp = nr.get("timestamp", 0)
            nr_input_msgs: list[Any] = nr.get("input_messages", [])
        else:
            nname = _html.escape(str(nr))
            nout = ""
            nr_tools, nr_final, nr_has_tools, nr_timestamp, nr_input_msgs = (
                [],
                False,
                False,
                0,
                [],
            )

        final_badge = (
            ' <span style="color:var(--color-pass);font-weight:600;">[FINAL]</span>'
            if nr_final
            else ""
        )
        tools_info = (
            f'<br/><span style="{_ms}">tools: {_html.escape(", ".join(nr_tools))}</span>'
            if nr_tools
            else ""
        )
        tc_flag = (
            f'<br/><span style="{_ms}">has_tool_calls: True</span>'
            if nr_has_tools
            else ""
        )
        ts_info = (
            f'<br/><span style="{_ms}">timestamp: {nr_timestamp}</span>'
            if nr_timestamp
            else ""
        )
        input_info = (
            f'<br/><span style="{_ms}">input_messages: {len(nr_input_msgs)} messages</span>'
            if nr_input_msgs
            else ""
        )

        boxes.append(
            f'<div class="node-box">'
            f'<span class="node-box-title">{nname}{final_badge}</span>'
            f"<p>{nout}</p>"
            f"{tools_info}{tc_flag}{ts_info}{input_info}"
            f"</div>"
        )
    return boxes


# ── Messages ──────────────────────────────────────────────────────────────────


def _render_messages(messages: list[Any]) -> list[str]:
    items: list[str] = []
    for msg in messages:
        role = msg.get("role", "?") if isinstance(msg, dict) else "?"
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        role_color = (
            "#6366f1"
            if role == "user"
            else "#22c55e"
            if role == "assistant"
            else "var(--color-muted)"
        )
        items.append(
            f'<div style="padding:0.25rem 0;font-size:0.8rem;">'
            f'<span style="color:{role_color};font-weight:600;">[{_html.escape(role)}]</span> '
            f"{_html.escape(content)}</div>"
        )
    return items


# ── Turn results ──────────────────────────────────────────────────────────────


def _render_turn_results(turn_results: list[Any]) -> list[str]:
    items: list[str] = []
    _ts = "color:var(--color-muted);font-size:0.75rem;"

    for tr in turn_results:
        tidx = tr.get("turn_index", "?")
        user_input = _html.escape(str(tr.get("user_input", "")))
        agent_resp = _html.escape(str(tr.get("agent_response", "")))
        turn_tcs: list[Any] = tr.get("tool_calls", [])
        turn_nv: list[str] = tr.get("node_visits", [])
        tc_info = (
            f'<br/><span style="{_ts}">tool_calls: {len(turn_tcs)}</span>'
            if turn_tcs
            else ""
        )
        nv_info = (
            f'<br/><span style="{_ts}">nodes: {_html.escape(" → ".join(turn_nv))}</span>'
            if turn_nv
            else ""
        )
        items.append(
            f'<div class="node-box">'
            f'<span class="node-box-title">Turn {tidx}</span>'
            f"<p><strong>User:</strong> {user_input}</p>"
            f"<p><strong>Agent:</strong> {agent_resp}</p>"
            f"{tc_info}{nv_info}"
            f"</div>"
        )
    return items
