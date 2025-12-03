"""
MCP Server for Oracle SQLcl with PTY support.

Windows에서 pywinpty를 사용하여 SQLcl 프로세스를 관리합니다.
PTY를 통해 버퍼링 문제를 해결하고 안정적인 통신을 제공합니다.
"""

import asyncio
import logging
import re
import time
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from winpty import PtyProcess

import sys
sys.path.insert(0, str(__file__).rsplit("\\", 2)[0])
from config import SQLCL_PATH, DB_CONNECTION, SQLCL_TIMEOUT

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('mcp_server.log', encoding='utf-8')]
)
logger = logging.getLogger("sqlcl-mcp-pty")

# MCP Server
app = Server("sqlcl-mcp")


class SQLclPTY:
    """PTY 기반 SQLcl 프로세스 관리자."""
    
    PROMPT_PATTERN = re.compile(r'SQL>\s*$', re.MULTILINE)
    END_MARKER = "__END_OF_RESULT__"
    
    def __init__(self):
        self.process: PtyProcess | None = None
        self.connected = False
        self._lock = asyncio.Lock()
    
    async def connect(self, connection: str = None) -> dict[str, Any]:
        """SQLcl에 연결합니다."""
        conn = connection or DB_CONNECTION
        if not conn:
            return {"success": False, "error": "데이터베이스 연결 정보가 필요합니다."}
        
        async with self._lock:
            try:
                if self.process and self.process.isalive():
                    return {"success": True, "message": "이미 연결됨"}
                
                # PTY 프로세스 시작
                logger.info(f"Starting SQLcl PTY: {SQLCL_PATH}")
                self.process = PtyProcess.spawn(
                    [SQLCL_PATH, "-S", conn],
                    dimensions=(100, 200)
                )
                
                # 프롬프트 대기
                await self._wait_for_prompt(timeout=30)
                
                # 초기 설정
                init_commands = [
                    "SET PAGESIZE 50000",
                    "SET LINESIZE 32767", 
                    "SET LONG 50000",
                    "SET TRIMSPOOL ON",
                    "SET TRIMOUT ON",
                    "SET FEEDBACK OFF",
                    "SET HEADING ON",
                    "SET SQLFORMAT csv",
                ]
                
                for cmd in init_commands:
                    await self._send_command(cmd)
                    await self._wait_for_prompt(timeout=5)
                
                self.connected = True
                logger.info("SQLcl PTY connected successfully")
                return {"success": True, "message": "연결 성공"}
                
            except Exception as e:
                logger.error(f"Connection failed: {e}")
                await self.disconnect()
                return {"success": False, "error": str(e)}
    
    async def disconnect(self) -> dict[str, Any]:
        """SQLcl 연결을 종료합니다."""
        async with self._lock:
            try:
                if self.process and self.process.isalive():
                    self.process.write("EXIT\r\n")
                    await asyncio.sleep(0.5)
                    self.process.terminate(force=True)
                self.process = None
                self.connected = False
                logger.info("SQLcl PTY disconnected")
                return {"success": True, "message": "연결 종료됨"}
            except Exception as e:
                logger.error(f"Disconnect error: {e}")
                return {"success": False, "error": str(e)}
    
    async def execute(self, sql: str) -> dict[str, Any]:
        """SQL을 실행하고 결과를 반환합니다."""
        if not self.connected or not self.process or not self.process.isalive():
            # 자동 재연결 시도
            result = await self.connect()
            if not result["success"]:
                return result
        
        async with self._lock:
            try:
                # SQL 정리
                sql = sql.strip()
                if not sql.endswith(';'):
                    sql += ';'
                
                # SQL 전송
                logger.debug(f"Executing: {sql[:100]}...")
                await self._send_command(sql)
                
                # 결과 수집
                output = await self._collect_output(timeout=SQLCL_TIMEOUT)
                
                # 결과 정리
                output = self._clean_output(output, sql)
                
                logger.debug(f"Result: {output[:200]}...")
                return {
                    "success": True,
                    "output": output
                }
                
            except asyncio.TimeoutError:
                logger.error("Execution timeout")
                return {"success": False, "error": "실행 시간 초과"}
            except Exception as e:
                logger.error(f"Execution error: {e}")
                # 연결 끊김 가능성 - 재연결 시도
                self.connected = False
                return {"success": False, "error": str(e)}
    
    async def _send_command(self, cmd: str):
        """명령어를 PTY에 전송합니다."""
        if self.process and self.process.isalive():
            self.process.write(cmd + "\r\n")
    
    async def _wait_for_prompt(self, timeout: float = 10):
        """SQL> 프롬프트가 나타날 때까지 대기합니다."""
        buffer = ""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.process and self.process.isalive():
                try:
                    data = self.process.read(1024, blocking=False)
                    if data:
                        buffer += data
                        if self.PROMPT_PATTERN.search(buffer):
                            return buffer
                except Exception:
                    pass
            await asyncio.sleep(0.05)
        
        raise asyncio.TimeoutError("Prompt wait timeout")
    
    async def _collect_output(self, timeout: float = 60) -> str:
        """실행 결과를 수집합니다."""
        buffer = ""
        start_time = time.time()
        no_data_count = 0
        
        while time.time() - start_time < timeout:
            if self.process and self.process.isalive():
                try:
                    data = self.process.read(4096, blocking=False)
                    if data:
                        buffer += data
                        no_data_count = 0
                        
                        # 프롬프트 발견 시 완료
                        if self.PROMPT_PATTERN.search(buffer):
                            return buffer
                    else:
                        no_data_count += 1
                        # 데이터가 없고 프롬프트가 있으면 완료
                        if no_data_count > 10 and self.PROMPT_PATTERN.search(buffer):
                            return buffer
                except Exception:
                    pass
            await asyncio.sleep(0.05)
        
        raise asyncio.TimeoutError("Output collection timeout")
    
    def _clean_output(self, output: str, original_sql: str) -> str:
        """출력에서 불필요한 부분을 제거합니다."""
        lines = output.split('\n')
        result_lines = []
        skip_echo = True
        
        for line in lines:
            # 에코된 SQL 명령 건너뛰기
            if skip_echo and original_sql[:30] in line:
                skip_echo = False
                continue
            
            # SQL> 프롬프트 제거
            if 'SQL>' in line:
                continue
            
            # ANSI 이스케이프 코드 제거
            clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            clean_line = re.sub(r'\x1b\[\?[0-9]*[a-z]', '', clean_line, flags=re.IGNORECASE)
            
            if clean_line.strip():
                result_lines.append(clean_line.rstrip())
        
        return '\n'.join(result_lines)


# 전역 SQLcl 인스턴스
sqlcl = SQLclPTY()


# =============================================================================
# MCP Tool Handlers
# =============================================================================

@app.list_tools()
async def list_tools() -> list[Tool]:
    """사용 가능한 도구 목록을 반환합니다."""
    return [
        Tool(
            name="connect",
            description="Oracle 데이터베이스에 연결합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "connection": {
                        "type": "string",
                        "description": "연결 문자열 (예: user/pass@host:port/service)"
                    }
                }
            }
        ),
        Tool(
            name="disconnect",
            description="데이터베이스 연결을 종료합니다.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="execute_sql",
            description="SQL 쿼리를 실행하고 결과를 반환합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "실행할 SQL 쿼리"
                    }
                },
                "required": ["sql"]
            }
        ),
        Tool(
            name="get_schema",
            description="테이블의 스키마 정보를 조회합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "조회할 테이블 이름"
                    }
                },
                "required": ["table_name"]
            }
        ),
        Tool(
            name="list_tables",
            description="현재 사용자의 테이블 목록을 조회합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "테이블 이름 패턴 (LIKE 조건, 선택사항)"
                    }
                }
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """도구를 실행합니다."""
    logger.info(f"Tool called: {name} with args: {arguments}")
    
    try:
        if name == "connect":
            result = await sqlcl.connect(arguments.get("connection"))
        
        elif name == "disconnect":
            result = await sqlcl.disconnect()
        
        elif name == "execute_sql":
            sql = arguments.get("sql", "")
            if not sql:
                result = {"success": False, "error": "SQL이 필요합니다."}
            else:
                result = await sqlcl.execute(sql)
        
        elif name == "get_schema":
            table_name = arguments.get("table_name", "").upper()
            if not table_name:
                result = {"success": False, "error": "테이블 이름이 필요합니다."}
            else:
                sql = f"""
                SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, NULLABLE
                FROM USER_TAB_COLUMNS
                WHERE TABLE_NAME = '{table_name}'
                ORDER BY COLUMN_ID
                """
                result = await sqlcl.execute(sql)
        
        elif name == "list_tables":
            pattern = arguments.get("pattern", "%")
            sql = f"""
            SELECT TABLE_NAME, NUM_ROWS
            FROM USER_TABLES
            WHERE TABLE_NAME LIKE '{pattern}'
            ORDER BY TABLE_NAME
            """
            result = await sqlcl.execute(sql)
        
        else:
            result = {"success": False, "error": f"알 수 없는 도구: {name}"}
        
        # 결과 포맷팅
        if result.get("success"):
            text = result.get("output", result.get("message", "완료"))
        else:
            text = f"오류: {result.get('error', '알 수 없는 오류')}"
        
        return [TextContent(type="text", text=text)]
        
    except Exception as e:
        logger.error(f"Tool error: {e}")
        return [TextContent(type="text", text=f"오류: {str(e)}")]


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """MCP 서버를 실행합니다."""
    logger.info("Starting SQLcl MCP PTY Server...")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
