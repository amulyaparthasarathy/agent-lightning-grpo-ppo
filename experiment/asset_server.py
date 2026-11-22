"""Real stdio MCP server. Protocol only on stdout; no harness environment required."""
import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult
try:
    from .asset_task import TOOL_SCHEMA, lookup, dumps, PolicyActionError
except ImportError:
    from asset_task import TOOL_SCHEMA, lookup, dumps, PolicyActionError

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
        return CallToolResult(isError=True, content=[TextContent(type='text', text=dumps({
            'error_type': 'asset_policy_action_v2', 'code': str(exc),
            'instruction': 'Call lookup_asset with exactly one string argument: asset_id.'}))])


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == '__main__':
    asyncio.run(main())
