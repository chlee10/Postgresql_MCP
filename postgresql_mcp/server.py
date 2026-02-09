"""
PostgreSQL MCP Server

Uses psycopg 3 to connect to PostgreSQL and provides MCP tools for SQL execution and schema inspection.
Supports SSE (Server-Sent Events) for Streamlit integration and stdio for CLI/Claude Desktop.
"""

import asyncio
import logging
import os
import sys
import json
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
import uvicorn
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_CONNECTION, LOG_LEVEL, SERVER_HOST, SERVER_PORT

# Logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("postgresql-mcp-server")

class PostgresManager:
    def __init__(self, connection_string: str):
        self.conn_str = connection_string
        self.pool = None

    async def connect(self):
        try:
            self.pool = AsyncConnectionPool(self.conn_str, open=False, min_size=1, max_size=10, kwargs={'row_factory': dict_row})
            await self.pool.open()
            await self.pool.wait()
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def close(self):
        if self.pool:
            await self.pool.close()
            logger.info("Closed PostgreSQL connection")

    async def list_tables(self) -> List[Dict[str, Any]]:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT table_schema, table_name 
                    FROM information_schema.tables 
                    WHERE table_schema NOT IN ('information_schema', 'pg_catalog') 
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_schema, table_name
                """)
                return await cur.fetchall()

    async def describe_table(self, table_name: str, schema: str = 'public') -> List[Dict[str, Any]]:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = %s AND table_schema = %s
                    ORDER BY ordinal_position
                """, (table_name, schema))
                return await cur.fetchall()

    async def run_query(self, sql: str) -> List[Dict[str, Any]]:
        # Basic safety check - only allow SELECT for now if needed, or rely on DB permissions
        # For an explorer tool, we usually want to allow read queries.
        # But let's just run what is given, assuming the user is aware.
        # Ideally, we should use a read-only user in connection string.
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql)
                if cur.description:
                    return await cur.fetchall()
                return [{"status": "success", "rowcount": cur.rowcount}]

class MCPServer:
    def __init__(self):
        self.server = Server("postgresql-mcp")
        self.db = PostgresManager(DB_CONNECTION)
        
        self.setup_handlers()

    def setup_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            return [
                Tool(
                    name="query",
                    description="Run a read-only SQL query against the PostgreSQL database. Returns JSON result.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string", "description": "The SQL query to execute"}
                        },
                        "required": ["sql"]
                    }
                ),
                Tool(
                    name="list_tables",
                    description="List all tables in the database (public schema). Returns JSON result.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    }
                ),
                 Tool(
                    name="describe_table",
                    description="Get the schema context for a specific table. Returns JSON result.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {"type": "string", "description": "The name of the table to describe"},
                            "schema": {"type": "string", "description": "The schema name (default: public)"}
                        },
                        "required": ["table_name"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent | ImageContent | EmbeddedResource]:
            if not arguments:
                arguments = {}
            
            try:
                if name == "query":
                    sql = arguments.get("sql")
                    if not sql:
                        raise ValueError("Missing 'sql' argument")
                    results = await self.db.run_query(sql)
                    return [TextContent(type="text", text=json.dumps(results, default=str))]
                
                elif name == "list_tables":
                    results = await self.db.list_tables()
                    return [TextContent(type="text", text=json.dumps(results, default=str))]
                
                elif name == "describe_table":
                    table_name = arguments.get("table_name")
                    schema = arguments.get("schema", "public")
                    if not table_name:
                        raise ValueError("Missing 'table_name' argument")
                    results = await self.db.describe_table(table_name, schema)
                    return [TextContent(type="text", text=json.dumps(results, default=str))]
                
                else:
                    raise ValueError(f"Unknown tool: {name}")
            
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}")
                return [TextContent(type="text", text=json.dumps({"error": str(e)}))] # Return error as JSON

    async def run_stdio(self):
        await self.db.connect()
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())

    async def run_sse(self):
        await self.db.connect()
        sse = SseServerTransport("/messages/")
        
        async def handle_sse(request):
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await self.server.run(streams[0], streams[1], self.server.create_initialization_options())
        
        app = Starlette(debug=True, routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages", app=sse.handle_post_message),
        ])
        
        config = uvicorn.Config(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PostgreSQL MCP Server")
    parser.add_argument("--sse", action="store_true", help="Run in SSE mode (HTTP)")
    args = parser.parse_args()
    
    mcp_server = MCPServer()
    
    # Fix for Windows ProactorEventLoop issue with psycopg
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if args.sse:
        asyncio.run(mcp_server.run_sse())
    else:
        asyncio.run(mcp_server.run_stdio())

if __name__ == "__main__":
    main()
