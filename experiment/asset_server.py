"""Real stdio MCP server. Protocol only on stdout; no harness environment required."""
import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
try:
    from .asset_task import TOOL_SCHEMA, lookup, dumps, policy_error, PolicyActionError
except ImportError:
    from asset_task import TOOL_SCHEMA, lookup, dumps, policy_error, PolicyActionError

server = Server('asset-status-v1')


@server.list_tools()
async def list_tools():
    return [Tool(name='lookup_asset', description='Return the synthetic service state and backup health of an asset.',
                 inputSchema=TOOL_SCHEMA)]


@server.call_tool(validate_input=False)
async def call_tool(name, arguments):
    try:
        if name != 'lookup_asset':
            raise PolicyActionError('unknown_tool')
        # Explicit validation also enforces no extra arguments (no coercion).
        return [TextContent(type='text', text=dumps(lookup(arguments)))]
    except PolicyActionError as exc:
        # MCP reports a raised tool exception as isError with exactly this text. That
        # conversion is the one observed live with the remote MCP version in a04.
        raise ValueError(dumps(policy_error(str(exc)))) from None


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == '__main__':
    asyncio.run(main())
