"""
SQLcl MCP Server - SSE/stdio 방식

스레드 기반 non-blocking I/O로 SQLcl과 안정적으로 통신
MCP 프로토콜을 통해 Claude Desktop, VS Code, Streamlit 등과 연동

사용법:
    SSE 모드 (기본): poetry run python -m sqlcl_mcp.server
    stdio 모드:      poetry run python -m sqlcl_mcp.server --stdio

버전: 2.1.0
"""

import asyncio
import logging
import os
import subprocess
import sys
import re
import threading
import queue
from typing import Optional

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent

# SSE/HTTP imports
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
import uvicorn

# Config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SQLCL_PATH, DB_CONNECTION, SQLCL_TIMEOUT, LOG_LEVEL, SERVER_HOST, SERVER_PORT

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('sqlcl_mcp_server.log'), logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("sqlcl-mcp-server")


# =============================================================================
# Non-blocking Reader
# =============================================================================
class NonBlockingReader:
    """Non-blocking stdout reader using a thread"""
    
    def __init__(self, stream):
        self.stream = stream
        self.output_queue: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._reader_thread, daemon=True)
        self.running = True
        self.thread.start()
    
    def _reader_thread(self):
        """Thread that reads from stream and puts into queue"""
        try:
            while self.running:
                line = self.stream.readline()
                if not line:
                    break
                self.output_queue.put(line)
        except Exception as e:
            logger.error(f"Reader thread error: {e}")
    
    def read_available(self, timeout: float = 0.1) -> str:
        """Read all available lines"""
        lines = []
        try:
            # 첫 번째 라인은 timeout으로
            line = self.output_queue.get(timeout=timeout)
            lines.append(line)
            
            # 나머지는 non-blocking으로
            while True:
                try:
                    line = self.output_queue.get_nowait()
                    lines.append(line)
                except queue.Empty:
                    break
        except queue.Empty:
            pass
        
        return ''.join(lines)
    
    def stop(self):
        """Stop the reader thread"""
        self.running = False


# =============================================================================
# SQLcl Persistent Session
# =============================================================================
class SQLclSession:
    """Subprocess 기반 SQLcl 세션 (스레드 기반 non-blocking I/O)"""
    
    def __init__(self, sqlcl_path: str, db_connection: str):
        self.sqlcl_path = sqlcl_path
        self.db_connection = db_connection
        self.process: Optional[subprocess.Popen] = None
        self.reader: Optional[NonBlockingReader] = None
        self.connected = False
        self._lock = asyncio.Lock()
        
    async def start(self) -> bool:
        """SQLcl 세션 시작"""
        try:
            # 환경 변수 설정
            env = os.environ.copy()
            env["NLS_LANG"] = "KOREAN_KOREA.AL32UTF8"
            env["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8"
            
            logger.info("Starting SQLcl session...")
            
            # Subprocess 시작
            self.process = subprocess.Popen(
                [self.sqlcl_path, "-S", self.db_connection],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,  # Line buffered
                env=env
            )
            
            logger.info(f"SQLcl process started, PID: {self.process.pid}")
            
            # Non-blocking reader 시작
            self.reader = NonBlockingReader(self.process.stdout)
            
            # 초기화 대기
            await asyncio.sleep(3)
            
            # 초기 출력 읽기 (로그인 메시지 등)
            initial = self.reader.read_available(1.0)
            logger.debug(f"Initial output: {initial[:200] if initial else 'None'}")
            
            # SQL 설정 명령 실행 - CSV format (헤더 없음, 후처리로 추가)
            init_commands = """SET PAGESIZE 50000
SET LINESIZE 32767
SET LONG 50000
SET FEEDBACK OFF
SET HEADING ON
SET VERIFY OFF
SET ECHO OFF
SET SQLFORMAT csv
"""
            self.process.stdin.write(init_commands)
            self.process.stdin.flush()
            
            await asyncio.sleep(1)
            self.reader.read_available(0.5)  # 설정 출력 버리기
            
            # 연결 테스트
            test_result = await self.execute("SELECT 'CONNECTED' AS STATUS FROM DUAL")
            if test_result and 'CONNECTED' in test_result:
                self.connected = True
                logger.info("[OK] SQLcl session connected!")
                return True
            else:
                logger.error(f"Connection test failed: {test_result}")
                return False
                
        except Exception as e:
            logger.exception(f"Failed to start SQLcl session: {e}")
            return False
    
    async def execute(self, sql: str, timeout: float = 60.0) -> str:
        """SQL 실행"""
        async with self._lock:
            if not self.process or not self.reader:
                return "ERROR: Session not started"
            
            if self.process.poll() is not None:
                return "ERROR: SQLcl process has terminated"
            
            try:
                sql = sql.strip()
                if not sql.endswith(';'):
                    sql += ';'
                
                # 이전 출력 비우기
                self.reader.read_available(0.1)
                
                # SQL 실행
                logger.debug(f"Executing: {sql[:100]}")
                self.process.stdin.write(sql + '\n')
                self.process.stdin.flush()
                
                # 결과 수집 (여러 번 시도)
                output_parts = []
                empty_count = 0
                start_time = asyncio.get_event_loop().time()
                
                while asyncio.get_event_loop().time() - start_time < timeout:
                    await asyncio.sleep(0.3)
                    data = self.reader.read_available(0.5)
                    
                    if data:
                        output_parts.append(data)
                        empty_count = 0
                    else:
                        empty_count += 1
                        if empty_count >= 3 and output_parts:
                            break
                
                # 결과 정리
                result = ''.join(output_parts)
                result = self._clean_output(result)
                
                logger.debug(f"Result: {result[:100] if result else 'None'}")
                return result
                
            except Exception as e:
                logger.exception(f"Execute error: {e}")
                return f"ERROR: {str(e)}"
    
    def _clean_output(self, output: str) -> str:
        """출력 정리"""
        lines = output.split('\n')
        clean_lines = []
        skip_patterns = [
            'Picked up JAVA_TOOL_OPTIONS',
            'SQL>',
        ]
        
        for line in lines:
            # 스킵 패턴
            if any(p in line for p in skip_patterns):
                continue
            
            # SQL 문장 번호 라인 제거 (예: "  2  SELECT ...", "  3* COUNT...", "  4* TO_CHAR...")
            stripped = line.lstrip()
            if stripped:
                # 숫자 또는 숫자*로 시작하는 SQL 스크립트 라인 스킵
                if re.match(r'^\d+\*?\s+', stripped):
                    # CSV 데이터가 아닌 경우만 스킵 (CSV는 따옴표로 시작하거나 숫자,숫자 형태)
                    rest = re.sub(r'^\d+\*?\s+', '', stripped)
                    if not rest.startswith('"') and not re.match(r'^[\d.]+,', rest):
                        continue
            
            # 구분선 제거 (___로만 이루어진 라인)
            if re.match(r'^[\s_]+$', line):
                continue
            
            # ANSI escape 코드 제거
            line = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', line)
            
            clean_lines.append(line)
        
        return '\n'.join(clean_lines).strip()
    
    async def close(self):
        """세션 종료"""
        if self.reader:
            self.reader.stop()
            self.reader = None
        
        if self.process:
            try:
                self.process.stdin.write("EXIT;\n")
                self.process.stdin.flush()
                await asyncio.sleep(0.5)
                self.process.terminate()
            except Exception:
                pass
            self.process = None
        
        self.connected = False
        logger.info("SQLcl session closed")


# =============================================================================
# Fallback: File-based execution
# =============================================================================
class SQLclFileExecutor:
    """파일 기반 SQLcl 실행 (Fallback)"""
    
    def __init__(self, sqlcl_path: str, db_connection: str):
        self.sqlcl_path = sqlcl_path
        self.db_connection = db_connection
        self.connected = False
        
    async def start(self) -> bool:
        """연결 테스트"""
        result = await self.execute("SELECT 'OK' FROM DUAL")
        self.connected = 'OK' in result
        return self.connected
    
    async def execute(self, sql: str, timeout: float = 60.0) -> str:
        """파일 기반 SQL 실행"""
        import tempfile
        
        sql = sql.strip()
        if not sql.endswith(';'):
            sql += ';'
        
        sql_content = f"""SET PAGESIZE 50000
SET LINESIZE 32767
SET LONG 50000
SET FEEDBACK OFF
SET HEADING ON
SET SQLFORMAT csv

{sql}

EXIT;
"""
        
        sql_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.sql', delete=False, encoding='utf-8'
            ) as f:
                f.write(sql_content)
                sql_file = f.name
            
            env = os.environ.copy()
            env["NLS_LANG"] = "KOREAN_KOREA.AL32UTF8"
            env["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8"
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    [self.sqlcl_path, "-S", self.db_connection, f"@{sql_file}"],
                    capture_output=True, text=True, timeout=timeout,
                    encoding='utf-8', errors='replace', env=env
                )
            )
            
            output = result.stdout.strip() if result.stdout else ""
            
            # JAVA_TOOL_OPTIONS 필터링
            lines = [
                line for line in output.split('\n') 
                if not line.startswith('Picked up JAVA_TOOL_OPTIONS')
            ]
            output = '\n'.join(lines).strip()
            
            if 'ORA-' in output or 'SP2-' in output:
                return f"ERROR: {output}"
            
            return output
            
        except subprocess.TimeoutExpired:
            return f"ERROR: Query timeout ({timeout}s)"
        except Exception as e:
            return f"ERROR: {str(e)}"
        finally:
            if sql_file and os.path.exists(sql_file):
                try:
                    os.remove(sql_file)
                except Exception:
                    pass
    
    async def close(self):
        self.connected = False


# =============================================================================
# MCP Server
# =============================================================================
# 전역 세션
sql_session: Optional[SQLclSession] = None
sql_fallback: Optional[SQLclFileExecutor] = None

# MCP 서버 인스턴스
server = Server("sqlcl-mcp")


@server.list_tools()
async def list_tools():
    """사용 가능한 도구 목록"""
    return [
        Tool(
            name="execute_sql",
            description="Execute an Oracle SQL query and return results in CSV format",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL query to execute"
                    }
                },
                "required": ["sql"]
            }
        ),
        Tool(
            name="get_tables",
            description="Get list of tables in the current schema",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="describe_table",
            description="Get structure of a specific table",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to describe"
                    }
                },
                "required": ["table_name"]
            }
        ),
        Tool(
            name="get_status",
            description="Get SQLcl connection status",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """도구 실행"""
    global sql_session, sql_fallback
    
    # 세션 확인
    session = sql_session if (sql_session and sql_session.connected) else sql_fallback
    
    if not session or not session.connected:
        return [TextContent(type="text", text="ERROR: Not connected to database")]
    
    try:
        if name == "execute_sql":
            sql = arguments.get("sql", "")
            if not sql:
                return [TextContent(type="text", text="ERROR: SQL query is required")]
            
            result = await session.execute(sql, timeout=SQLCL_TIMEOUT)
            return [TextContent(type="text", text=result)]
        
        elif name == "get_tables":
            sql = "SELECT table_name FROM user_tables ORDER BY table_name"
            result = await session.execute(sql)
            return [TextContent(type="text", text=result)]
        
        elif name == "describe_table":
            table_name = arguments.get("table_name", "").upper()
            if not table_name:
                return [TextContent(type="text", text="ERROR: table_name is required")]
            
            sql = f"""SELECT column_name, data_type, nullable, data_length 
                      FROM user_tab_columns 
                      WHERE table_name = '{table_name}' 
                      ORDER BY column_id"""
            result = await session.execute(sql)
            return [TextContent(type="text", text=result)]
        
        elif name == "get_status":
            mode = "Persistent" if (sql_session and sql_session.connected) else "File"
            status = f"Connected: {session.connected}\nMode: {mode}"
            return [TextContent(type="text", text=status)]
        
        else:
            return [TextContent(type="text", text=f"ERROR: Unknown tool: {name}")]
            
    except Exception as e:
        logger.exception(f"Tool error: {e}")
        return [TextContent(type="text", text=f"ERROR: {str(e)}")]


async def initialize():
    """서버 초기화"""
    global sql_session, sql_fallback
    
    logger.info("=" * 60)
    logger.info("SQLcl MCP Server Starting...")
    logger.info("=" * 60)
    logger.info(f"SQLcl Path: {SQLCL_PATH}")
    logger.info(f"DB Connection: {'Set' if DB_CONNECTION else 'NOT SET!'}")
    
    if not DB_CONNECTION:
        logger.error("DB_CONNECTION not set!")
        return
    
    # Persistent 세션 시도
    logger.info("Trying persistent session mode...")
    sql_session = SQLclSession(SQLCL_PATH, DB_CONNECTION)
    
    if await sql_session.start():
        logger.info("[OK] Persistent session active!")
    else:
        logger.warning("Persistent session failed, falling back to file mode...")
        sql_session = None
        
        # Fallback: 파일 방식
        sql_fallback = SQLclFileExecutor(SQLCL_PATH, DB_CONNECTION)
        if await sql_fallback.start():
            logger.info("[OK] File mode active!")
        else:
            logger.error("[FAIL] All connection modes failed!")


async def cleanup():
    """정리"""
    global sql_session, sql_fallback
    
    if sql_session:
        await sql_session.close()
    if sql_fallback:
        await sql_fallback.close()
    
    logger.info("Server shutdown complete")


# =============================================================================
# SSE Server (HTTP 방식 - 상주 서버)
# =============================================================================
sse_transport = SseServerTransport("/messages")

async def handle_status(request):
    """상태 확인"""
    session = sql_session if (sql_session and sql_session.connected) else sql_fallback
    return JSONResponse({
        "status": "running",
        "connected": session.connected if session else False,
        "mode": "persistent" if (sql_session and sql_session.connected) else "file"
    })

async def lifespan(app):
    """앱 시작/종료"""
    await initialize()
    yield
    await cleanup()

class SSEApp:
    """SSE ASGI 앱 - connect_sse와 handle_post_message 처리"""
    
    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        
        if path == "/sse" and method == "GET":
            # SSE 연결
            async with sse_transport.connect_sse(scope, receive, send) as streams:
                await server.run(
                    streams[0], 
                    streams[1], 
                    server.create_initialization_options()
                )
        elif path == "/messages" and method == "POST":
            # POST 메시지 처리
            await sse_transport.handle_post_message(scope, receive, send)
        else:
            # 404
            await send({
                "type": "http.response.start",
                "status": 404,
                "headers": [[b"content-type", b"text/plain"]],
            })
            await send({
                "type": "http.response.body",
                "body": b"Not Found",
            })

def create_sse_app():
    """SSE 앱 생성"""
    from starlette.routing import Mount
    
    return Starlette(
        routes=[
            Route("/status", endpoint=handle_status),
            Mount("/", app=SSEApp()),
        ],
        lifespan=lifespan
    )


async def main_stdio():
    """stdio 모드 (기존)"""
    await initialize()
    
    logger.info("Starting MCP stdio server...")
    
    async with stdio_server() as (read_stream, write_stream):
        try:
            await server.run(read_stream, write_stream, server.create_initialization_options())
        finally:
            await cleanup()


def main_sse():
    """SSE 모드 (HTTP 상주 서버)"""
    host = SERVER_HOST
    port = SERVER_PORT
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║         SQLcl MCP Server (SSE Mode)                      ║
╠══════════════════════════════════════════════════════════╣
║  Server: http://{host}:{port}                            ║
║                                                          ║
║  Endpoints:                                              ║
║    GET  /sse      - SSE connection for MCP               ║
║    POST /messages - MCP message handler                  ║
║    GET  /status   - Server status                        ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    app = create_sse_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    # --stdio 옵션으로 stdio 모드 실행 (Claude Desktop 연동용)
    if "--stdio" in sys.argv:
        try:
            asyncio.run(main_stdio())
        except KeyboardInterrupt:
            logger.info("Server interrupted")
        except Exception as e:
            logger.exception(f"Server error: {e}")
        sys.exit(1)
    else:
        # 기본: SSE 모드 (Streamlit UI 사용)
        main_sse()
