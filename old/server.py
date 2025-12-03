"""MCP Server for Oracle SQLcl integration."""

import asyncio
import logging
import os
import subprocess
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sqlcl-mcp")

# 환경 변수에서 설정 읽기
SQLCL_PATH = os.getenv("SQLCL_PATH", "sql")
DB_CONNECTION = os.getenv("DB_CONNECTION", "")

# MCP 서버 초기화
app = Server("sqlcl-mcp")


def execute_sqlcl(sql_command: str, connection: str = None) -> dict[str, Any]:
    """
    SQLcl을 실행하고 결과를 반환합니다.
    
    Args:
        sql_command: 실행할 SQL 명령어
        connection: 데이터베이스 연결 문자열 (옵션)
    
    Returns:
        실행 결과를 포함하는 딕셔너리
    """
    conn = connection or DB_CONNECTION
    
    if not conn:
        return {
            "success": False,
            "error": "데이터베이스 연결 정보가 제공되지 않았습니다."
        }
    
    try:
        # SQLcl 명령어 구성
        # -S: Silent mode, -L: 로그온만, nolog: 연결 없이 시작
        cmd = [
            SQLCL_PATH,
            "-S",  # Silent mode
            conn,
        ]
        
        # SQL 명령어에 exit 추가
        full_sql = f"SET PAGESIZE 0\nSET FEEDBACK OFF\nSET HEADING ON\n{sql_command}\nEXIT;"
        
        # SQLcl 실행
        result = subprocess.run(
            cmd,
            input=full_sql,
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            return {
                "success": True,
                "output": result.stdout.strip(),
                "error": result.stderr.strip() if result.stderr else None
            }
        else:
            return {
                "success": False,
                "error": f"SQLcl 실행 실패 (코드: {result.returncode})",
                "output": result.stdout.strip() if result.stdout else None,
                "stderr": result.stderr.strip() if result.stderr else None
            }
            
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "SQL 실행 시간 초과 (30초)"
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": f"SQLcl 실행 파일을 찾을 수 없습니다: {SQLCL_PATH}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"예상치 못한 오류: {str(e)}"
        }


@app.list_tools()
async def list_tools() -> list[Tool]:
    """사용 가능한 도구 목록을 반환합니다."""
    return [
        Tool(
            name="execute_sql",
            description="Oracle 데이터베이스에서 SQL 쿼리를 실행합니다. SELECT, INSERT, UPDATE, DELETE 등 모든 SQL 문을 지원합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "실행할 SQL 쿼리 (예: SELECT * FROM employees WHERE department_id = 10)"
                    },
                    "connection": {
                        "type": "string",
                        "description": "데이터베이스 연결 문자열 (옵션, 형식: username/password@host:port/service)"
                    }
                },
                "required": ["sql"]
            }
        ),
        Tool(
            name="describe_table",
            description="Oracle 테이블의 구조를 조회합니다 (컬럼명, 데이터 타입, NULL 허용 여부 등).",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "조회할 테이블명"
                    },
                    "connection": {
                        "type": "string",
                        "description": "데이터베이스 연결 문자열 (옵션)"
                    }
                },
                "required": ["table_name"]
            }
        ),
        Tool(
            name="list_tables",
            description="현재 사용자가 접근 가능한 모든 테이블 목록을 조회합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "connection": {
                        "type": "string",
                        "description": "데이터베이스 연결 문자열 (옵션)"
                    }
                },
                "required": []
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """도구 호출을 처리합니다."""
    
    if name == "execute_sql":
        sql = arguments.get("sql")
        connection = arguments.get("connection")
        
        if not sql:
            return [TextContent(
                type="text",
                text="오류: SQL 쿼리가 제공되지 않았습니다."
            )]
        
        result = execute_sqlcl(sql, connection)
        
        if result["success"]:
            output = result["output"] or "쿼리가 성공적으로 실행되었습니다."
            return [TextContent(type="text", text=output)]
        else:
            error_msg = f"오류: {result['error']}"
            if result.get("stderr"):
                error_msg += f"\n상세: {result['stderr']}"
            return [TextContent(type="text", text=error_msg)]
    
    elif name == "describe_table":
        table_name = arguments.get("table_name")
        connection = arguments.get("connection")
        
        if not table_name:
            return [TextContent(
                type="text",
                text="오류: 테이블명이 제공되지 않았습니다."
            )]
        
        sql = f"DESC {table_name}"
        result = execute_sqlcl(sql, connection)
        
        if result["success"]:
            return [TextContent(type="text", text=result["output"])]
        else:
            return [TextContent(type="text", text=f"오류: {result['error']}")]
    
    elif name == "list_tables":
        connection = arguments.get("connection")
        
        sql = "SELECT table_name FROM user_tables ORDER BY table_name;"
        result = execute_sqlcl(sql, connection)
        
        if result["success"]:
            return [TextContent(type="text", text=result["output"])]
        else:
            return [TextContent(type="text", text=f"오류: {result['error']}")]
    
    else:
        return [TextContent(
            type="text",
            text=f"오류: 알 수 없는 도구입니다: {name}"
        )]


async def main():
    """MCP 서버를 실행합니다."""
    logger.info("SQLcl MCP 서버를 시작합니다...")
    logger.info(f"SQLcl 경로: {SQLCL_PATH}")
    logger.info(f"DB 연결 설정: {'있음' if DB_CONNECTION else '없음 (도구 호출 시 제공 필요)'}")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
