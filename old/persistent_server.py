"""
Persistent MCP Server for Oracle SQLcl integration.

이 서버는 SQLcl 세션을 Persistent하게 유지하여 매 쿼리마다 로그인하지 않습니다.
- 서버 시작시 1회만 DB 로그인
- 이후 쿼리는 기존 세션에서 실행 (0.1~0.5초)
- 연결 끊김시 자동 재연결
"""

import asyncio
import logging
import os
import time
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sqlcl-mcp-persistent")

# 환경 변수에서 설정 읽기
SQLCL_PATH = os.getenv("SQLCL_PATH", "sql")
DB_CONNECTION = os.getenv("DB_CONNECTION", "")

# MCP 서버 초기화
app = Server("sqlcl-mcp-persistent")


class PersistentSQLclSession:
    """
    SQLcl 프로세스를 Persistent하게 유지하는 세션 클래스.
    
    서버 시작시 1회 로그인 후, 이후 쿼리는 stdin/stdout으로 통신합니다.
    """
    
    def __init__(self, sqlcl_path: str, db_connection: str):
        self.sqlcl_path = sqlcl_path
        self.db_connection = db_connection
        self.process: Optional[asyncio.subprocess.Process] = None
        self.connected = False
        self._lock = asyncio.Lock()
        self._response_marker = "===QUERY_COMPLETE==="
        self._error_marker = "===QUERY_ERROR==="
        
    def _get_env(self) -> dict:
        """SQLcl 실행에 필요한 환경 변수를 반환합니다."""
        env = os.environ.copy()
        env["NLS_LANG"] = "KOREAN_KOREA.AL32UTF8"
        env["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8"
        return env
    
    async def connect(self) -> bool:
        """DB에 연결합니다. 서버 시작시 1회만 호출됩니다."""
        if self.connected and self.process and self.process.returncode is None:
            logger.info("Already connected")
            return True
        
        try:
            logger.info("Connecting to database...")
            logger.info(f"SQLcl path: {self.sqlcl_path}")
            
            # SQLcl 프로세스 시작
            self.process = await asyncio.create_subprocess_exec(
                self.sqlcl_path,
                "-S",  # Silent mode
                self.db_connection,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_env()
            )
            
            # 초기 설정 명령 전송
            init_commands = """
SET PAGESIZE 50000
SET LINESIZE 32767
SET LONG 50000
SET LONGCHUNKSIZE 50000
SET TRIMSPOOL ON
SET TRIMOUT ON
SET FEEDBACK OFF
SET HEADING ON
SET SQLFORMAT csv
SET SERVEROUTPUT ON
"""
            await self._send_command(init_commands)
            
            # 연결 테스트
            success, result = await self.execute("SELECT 'CONNECTION_OK' AS STATUS FROM DUAL")
            
            if success and 'CONNECTION_OK' in result:
                self.connected = True
                logger.info("✅ Database connection successful!")
                return True
            else:
                logger.error(f"❌ Connection test failed: {result}")
                await self.disconnect()
                return False
                
        except FileNotFoundError:
            logger.error(f"❌ SQLcl not found: {self.sqlcl_path}")
            return False
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            await self.disconnect()
            return False
    
    async def disconnect(self):
        """연결을 종료합니다."""
        if self.process:
            try:
                self.process.stdin.write(b"EXIT;\n")
                await self.process.stdin.drain()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except Exception:
                self.process.kill()
            finally:
                self.process = None
                self.connected = False
                logger.info("Disconnected from database")
    
    async def _send_command(self, command: str):
        """명령을 SQLcl 프로세스에 전송합니다."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("Process not running")
        
        self.process.stdin.write(command.encode('utf-8'))
        await self.process.stdin.drain()
    
    async def _read_until_marker(self, timeout: float = 60.0) -> tuple[bool, str]:
        """마커가 나올 때까지 출력을 읽습니다."""
        output_lines = []
        start_time = time.time()
        
        try:
            while True:
                if time.time() - start_time > timeout:
                    return False, "Query timeout"
                
                try:
                    line = await asyncio.wait_for(
                        self.process.stdout.readline(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                if not line:
                    # 프로세스 종료됨
                    self.connected = False
                    return False, "Process terminated unexpectedly"
                
                decoded = line.decode('utf-8', errors='replace').rstrip('\r\n')
                
                # 마커 체크
                if self._response_marker in decoded:
                    break
                if self._error_marker in decoded:
                    return False, '\n'.join(output_lines)
                
                # JAVA_TOOL_OPTIONS 메시지 필터링
                if decoded.startswith('Picked up JAVA_TOOL_OPTIONS'):
                    continue
                
                output_lines.append(decoded)
            
            return True, '\n'.join(output_lines)
            
        except Exception as e:
            return False, f"Read error: {str(e)}"
    
    async def execute(self, sql: str, timeout: float = 60.0) -> tuple[bool, str]:
        """
        SQL 쿼리를 실행합니다.
        
        이미 로그인된 세션에서 실행하므로 매우 빠릅니다.
        """
        async with self._lock:
            # 연결 상태 확인
            if not self.connected or not self.process or self.process.returncode is not None:
                logger.warning("Connection lost, attempting to reconnect...")
                if not await self.connect():
                    return False, "Failed to reconnect to database"
            
            sql = sql.strip()
            if not sql.endswith(';'):
                sql += ';'
            
            start_time = time.time()
            logger.info(f"Executing query: {sql[:100]}...")
            
            try:
                # 쿼리 실행 및 마커 출력
                command = f"""
{sql}
PROMPT {self._response_marker}
"""
                await self._send_command(command)
                
                # 결과 읽기
                success, output = await self._read_until_marker(timeout)
                
                elapsed = time.time() - start_time
                logger.info(f"Query completed in {elapsed:.3f}s")
                
                # 결과 정리
                output = self._clean_output(output)
                
                # 에러 체크
                if self._has_error(output):
                    return False, output
                
                return success, output
                
            except Exception as e:
                logger.error(f"Execution error: {e}")
                return False, f"Execution error: {str(e)}"
    
    def _clean_output(self, output: str) -> str:
        """출력을 정리합니다."""
        lines = output.split('\n')
        cleaned = []
        
        for line in lines:
            # 빈 줄 스킵 (맨 앞/뒤)
            stripped = line.strip()
            
            # 메타 정보 스킵
            if stripped in ['', 'Commit complete.', 'Rollback complete.']:
                if cleaned:  # 중간 빈 줄은 유지
                    cleaned.append(line)
                continue
            
            # PROMPT 마커 스킵
            if self._response_marker in stripped or self._error_marker in stripped:
                continue
            
            cleaned.append(line)
        
        # 앞뒤 빈 줄 제거
        while cleaned and not cleaned[0].strip():
            cleaned.pop(0)
        while cleaned and not cleaned[-1].strip():
            cleaned.pop()
        
        return '\n'.join(cleaned)
    
    def _has_error(self, output: str) -> bool:
        """에러 메시지가 있는지 확인합니다."""
        error_patterns = ['ORA-', 'SP2-', 'Error at', 'PLS-']
        upper_output = output.upper()
        return any(pattern.upper() in upper_output for pattern in error_patterns)
    
    def is_connected(self) -> bool:
        """연결 상태를 반환합니다."""
        return self.connected and self.process is not None and self.process.returncode is None


# 글로벌 세션 인스턴스
session: Optional[PersistentSQLclSession] = None


async def get_session() -> PersistentSQLclSession:
    """싱글톤 세션을 반환합니다."""
    global session
    
    if session is None:
        session = PersistentSQLclSession(SQLCL_PATH, DB_CONNECTION)
    
    if not session.is_connected():
        await session.connect()
    
    return session


@app.list_tools()
async def list_tools() -> list[Tool]:
    """사용 가능한 도구 목록을 반환합니다."""
    return [
        Tool(
            name="execute_sql",
            description="Oracle 데이터베이스에서 SQL 쿼리를 실행합니다. SELECT, INSERT, UPDATE, DELETE 등 모든 SQL 문을 지원합니다. Persistent 연결을 사용하여 빠르게 실행됩니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "실행할 SQL 쿼리 (예: SELECT * FROM employees WHERE department_id = 10)"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "쿼리 타임아웃 (초), 기본값 60초"
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
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="connection_status",
            description="현재 데이터베이스 연결 상태를 확인합니다.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="reconnect",
            description="데이터베이스 연결을 재설정합니다.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """도구 호출을 처리합니다."""
    global session
    
    if name == "execute_sql":
        sql = arguments.get("sql")
        timeout = arguments.get("timeout", 60)
        
        if not sql:
            return [TextContent(
                type="text",
                text="오류: SQL 쿼리가 제공되지 않았습니다."
            )]
        
        try:
            sess = await get_session()
            success, result = await sess.execute(sql, timeout=timeout)
            
            if success:
                output = result or "쿼리가 성공적으로 실행되었습니다."
                return [TextContent(type="text", text=output)]
            else:
                return [TextContent(type="text", text=f"오류: {result}")]
        except Exception as e:
            return [TextContent(type="text", text=f"실행 오류: {str(e)}")]
    
    elif name == "describe_table":
        table_name = arguments.get("table_name")
        
        if not table_name:
            return [TextContent(
                type="text",
                text="오류: 테이블명이 제공되지 않았습니다."
            )]
        
        try:
            sess = await get_session()
            success, result = await sess.execute(f"DESC {table_name}")
            
            if success:
                return [TextContent(type="text", text=result)]
            else:
                return [TextContent(type="text", text=f"오류: {result}")]
        except Exception as e:
            return [TextContent(type="text", text=f"실행 오류: {str(e)}")]
    
    elif name == "list_tables":
        try:
            sess = await get_session()
            success, result = await sess.execute(
                "SELECT table_name FROM user_tables ORDER BY table_name"
            )
            
            if success:
                return [TextContent(type="text", text=result)]
            else:
                return [TextContent(type="text", text=f"오류: {result}")]
        except Exception as e:
            return [TextContent(type="text", text=f"실행 오류: {str(e)}")]
    
    elif name == "connection_status":
        if session and session.is_connected():
            db_info = DB_CONNECTION.split('@')[1] if '@' in DB_CONNECTION else 'Unknown'
            return [TextContent(
                type="text",
                text=f"✅ 연결됨\n데이터베이스: {db_info}"
            )]
        else:
            return [TextContent(
                type="text",
                text="❌ 연결되지 않음"
            )]
    
    elif name == "reconnect":
        try:
            if session:
                await session.disconnect()
            
            session = PersistentSQLclSession(SQLCL_PATH, DB_CONNECTION)
            success = await session.connect()
            
            if success:
                return [TextContent(
                    type="text",
                    text="✅ 재연결 성공"
                )]
            else:
                return [TextContent(
                    type="text",
                    text="❌ 재연결 실패"
                )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"❌ 재연결 오류: {str(e)}"
            )]
    
    else:
        return [TextContent(
            type="text",
            text=f"오류: 알 수 없는 도구입니다: {name}"
        )]


async def main():
    """MCP 서버를 실행합니다."""
    global session
    
    logger.info("=" * 60)
    logger.info("SQLcl Persistent MCP Server 시작")
    logger.info("=" * 60)
    logger.info(f"SQLcl 경로: {SQLCL_PATH}")
    logger.info(f"DB 연결: {'설정됨' if DB_CONNECTION else '없음'}")
    
    # 서버 시작시 DB 연결
    if DB_CONNECTION:
        logger.info("초기 DB 연결 시도...")
        session = PersistentSQLclSession(SQLCL_PATH, DB_CONNECTION)
        connected = await session.connect()
        
        if connected:
            logger.info("✅ 초기 DB 연결 성공!")
        else:
            logger.warning("⚠️ 초기 DB 연결 실패 - 첫 쿼리 시 재시도합니다")
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    finally:
        # 서버 종료시 연결 정리
        if session:
            await session.disconnect()
            logger.info("DB 연결 종료됨")


if __name__ == "__main__":
    asyncio.run(main())
