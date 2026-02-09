"""
PostgreSQL MCP Server

psycopg 3 기반 PostgreSQL 연결을 통해 MCP(Model Context Protocol) 도구를 제공합니다.

지원 전송 방식:
  - stdio  : Claude Desktop 등 MCP 클라이언트와 직접 통신 (기본 모드)
  - SSE    : Streamlit 등 웹 클라이언트와 HTTP Server-Sent Events 통신 (--sse 옵션)

제공 도구(Tools):
  - query          : SQL 쿼리 실행 (결과를 JSON으로 반환)
  - list_tables    : public 스키마의 테이블 목록 조회
  - describe_table : 특정 테이블의 컬럼 구조 조회
"""

import asyncio
import logging
import os
import sys
import json
from typing import List, Dict, Any

# --- MCP SDK ---
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

# --- Web (SSE 모드 전용) ---
from starlette.applications import Starlette
from starlette.routing import Route, Mount
import uvicorn

# --- PostgreSQL ---
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


# ---------------------------------------------------------------------------
# Windows 환경: stdin/stdout UTF-8 강제 설정
# ---------------------------------------------------------------------------
if sys.platform == 'win32':
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

# 상위 디렉토리(config.py 위치)를 import 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_CONNECTION, LOG_LEVEL, SERVER_HOST, SERVER_PORT

# ---------------------------------------------------------------------------
# 로깅 설정
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("postgresql-mcp-server")

# ===========================================================================
# PostgresManager — 비동기 커넥션 풀 기반 DB 접근 계층
# ===========================================================================
class PostgresManager:
    """psycopg_pool.AsyncConnectionPool 을 래핑하여 DB 조회 메서드를 제공합니다."""

    def __init__(self, connection_string: str):
        self.conn_str = connection_string
        self.pool: AsyncConnectionPool | None = None

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
        logger.info(f"Describing table: {table_name}, schema: {schema}")
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns 
                        WHERE table_name = %s AND table_schema = %s
                        ORDER BY ordinal_position
                    """, (table_name, schema))
                    res = await cur.fetchall()
                    logger.info(f"Describe table fetched {len(res)} columns")
                    return res
        except Exception as e:
            logger.error(f"Error describing table {table_name}: {e}")
            raise

    async def run_query(self, sql: str) -> List[Dict[str, Any]]:
        """임의의 SQL 실행. 보안을 위해 DB 사용자를 읽기 전용으로 설정하는 것을 권장합니다."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql)
                if cur.description:
                    return await cur.fetchall()
                return [{"status": "success", "rowcount": cur.rowcount}]

# ===========================================================================
# MCPServer — MCP 프로토콜 도구 등록 및 서버 실행
# ===========================================================================
class MCPServer:
    """MCP Server 인스턴스. stdio 또는 SSE 모드로 실행할 수 있습니다."""

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

    # ----- 전송 모드: stdio (Claude Desktop 등) -----
    async def run_stdio(self):
        """stdio 모드로 서버 실행 (Claude Desktop, CLI 클라이언트용)."""
        await self.db.connect()
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, write_stream,
                self.server.create_initialization_options()
            )

    # ----- 전송 모드: SSE (Streamlit 등 웹 클라이언트) -----
    async def run_sse(self):
        """SSE(Server-Sent Events) 모드로 서버 실행 (Streamlit 웹 UI용)."""
        await self.db.connect()
        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await self.server.run(
                    streams[0], streams[1],
                    self.server.create_initialization_options()
                )

        app = Starlette(debug=True, routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages", app=sse.handle_post_message),
        ])

        config = uvicorn.Config(
            app, host=SERVER_HOST, port=SERVER_PORT, log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

# ===========================================================================
# 엔트리포인트
# ===========================================================================
def main():
    """CLI 엔트리포인트. --sse 옵션으로 SSE 모드, 기본은 stdio 모드."""
    import argparse

    parser = argparse.ArgumentParser(description="PostgreSQL MCP Server")
    parser.add_argument("--sse", action="store_true", help="SSE(HTTP) 모드로 실행")
    args = parser.parse_args()

    # Windows: ProactorEventLoop 이슈 우회 (psycopg 호환)
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    mcp_server = MCPServer()

    if args.sse:
        asyncio.run(mcp_server.run_sse())
    else:
        asyncio.run(mcp_server.run_stdio())


if __name__ == "__main__":
    main()
