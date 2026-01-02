# Agentflow Release Notes
**PyPI Project:** [10xscale-agentflow v0.5.6](https://pypi.org/project/10xscale-agentflow/0.5.6/)

## Version: 0.5.6 (later)
**Summary:**
This release brings major improvements to error handling, tool execution, agent creation, and database robustness. All notable changes and fixes from versions 0.5.1 through 0.5.5 are consolidated here for clarity and migration ease.

### Key Enhancements
- Improved error handling and retry logic in PgCheckpointer, with actionable logging and safer thread updates (preserving thread names and returning creation status).
- Enhanced interrupt handling for MCP and Node execution, ensuring predictable pause/resume and correct event metadata.
- MCP Tool Execution now supports passing user information for richer context.
- Introduced the Agent class for simplified agent instantiation and LLM integration.
- Added tag-based tool filtering and @tool decorator for better tool discovery and schema generation.
- Support for handoff tools and remote tools, enabling agent transfer and external tool call handling.
- Parallel tool calls within nodes for improved throughput.
- ID generation and SQL type mapping fixes for numeric types (int, bigint).
- Improved logging with module-specific logger names.
- Fixed timestamp handling and thread reconstruction in PgCheckpointer, including updated_at field for accurate thread snapshots.

### Impact & Migration Notes
- PgCheckpointer API now protects against NULL thread name updates and provides clearer error propagation for diagnostics.
- Interrupt handling is more deterministic; users relying on pause/resume should see improved behavior.
- To forward user details to MCP tools, instantiate ToolNode with pass_user_info_to_mcp=True (opt-in).
- Use tags to filter tools in agents and tool nodes; remote tool behavior requires frontends to handle RemoteToolCallBlock messages and resume graphs appropriately.
- Parallel tool execution may improve performance for workflows with multiple tool calls.
- Historical data and UI displays now incorporate updated_at for threads, improving ordering and accuracy.
- No breaking API changes, but migration is recommended to leverage new features and robustness improvements.
