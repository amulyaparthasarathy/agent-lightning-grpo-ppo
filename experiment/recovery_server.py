"""Real stdio MCP server for recovery-readiness-v1. Protocol only on stdout.

Three read-only tools over the regenerated deterministic world. No data file,
label, case family, harness environment or wall-clock time is read.
"""
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
try:
    from .recovery_task import TOOLS, TOOL_DESCRIPTIONS, TOOL_SCHEMAS, call, dumps, policy_error, PolicyActionError
except ImportError:
    from recovery_task import TOOLS, TOOL_DESCRIPTIONS, TOOL_SCHEMAS, call, dumps, policy_error, PolicyActionError

server = Server('recovery-readiness-v1')


@server.list_tools()
async def list_tools():
    return [Tool(name=name, description=TOOL_DESCRIPTIONS[name], inputSchema=TOOL_SCHEMAS[name]) for name in TOOLS]


@server.call_tool(validate_input=False)
async def call_tool(name, arguments):
    try:
        # Explicit validation also enforces no extra arguments (no coercion or repair).
        return [TextContent(type='text', text=dumps(call(name, arguments)))]
    except PolicyActionError as exc:
        # MCP reports a raised tool exception as isError with exactly this text.
        raise ValueError(dumps(policy_error(str(exc)))) from None


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == '__main__':
    asyncio.run(main())
