"""
Standalone Persistent DB Server (HTTP 기반)

이 서버를 먼저 실행하면:
1. 서버 시작시 DB 로그인 완료
2. Streamlit 앱은 이미 로그인된 서버에 연결
3. 첫 쿼리부터 빠른 응답 (0.1~0.2초)

실행 방법:
    python -m sqlcl_mcp.db_server

또는:
    python sqlcl_mcp/db_server.py
"""

import asyncio
import logging
import os
import time
from typing import Optional
from dotenv import load_dotenv
from aiohttp import web

# Load environment variables first
load_dotenv(override=True)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db-server")

# 환경 변수에서 설정 읽기
SQLCL_PATH = os.getenv("SQLCL_PATH", "sql")
DB_CONNECTION = os.getenv("DB_CONNECTION", "")
SERVER_HOST = os.getenv("DB_SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("DB_SERVER_PORT", "8765"))


class PersistentSQLclSession:
    """
    SQLcl 프로세스를 Persistent하게 유지하는 세션 클래스.
    """
    
    def __init__(self, sqlcl_path: str, db_connection: str):
        self.sqlcl_path = sqlcl_path
        self.db_connection = db_connection
        self.process: Optional[asyncio.subprocess.Process] = None
        self.connected = False
        self._lock = asyncio.Lock()
        self._response_marker = "===QUERY_COMPLETE_12345==="
        self._query_count = 0
        self._total_query_time = 0.0
        self._connection_time = None
        
    def _get_env(self) -> dict:
        """SQLcl 실행에 필요한 환경 변수를 반환합니다."""
        env = os.environ.copy()
        env["NLS_LANG"] = "KOREAN_KOREA.AL32UTF8"
        env["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8"
        return env
    
    async def connect(self) -> bool:
        """DB에 연결합니다."""
        if self.connected and self.process and self.process.returncode is None:
            logger.info("Already connected")
            return True
        
        try:
            logger.info("=" * 50)
            logger.info("Connecting to database...")
            logger.info(f"SQLcl path: {self.sqlcl_path}")
            
            start_time = time.time()
            
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
            self.process.stdin.write(init_commands.encode('utf-8'))
            await self.process.stdin.drain()
            
            # 짧은 대기
            await asyncio.sleep(0.5)
            
            # 연결 테스트
            success, result = await self.execute("SELECT 'CONNECTION_OK' AS STATUS FROM DUAL")
            
            elapsed = time.time() - start_time
            
            if success and 'CONNECTION_OK' in result:
                self.connected = True
                self._connection_time = time.time()
                logger.info("=" * 50)
                logger.info(f"✅ Database connected! (took {elapsed:.2f}s)")
                logger.info("=" * 50)
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
    
    async def execute(self, sql: str, timeout: float = 60.0) -> tuple[bool, str]:
        """SQL 쿼리를 실행합니다."""
        async with self._lock:
            if not self.connected or not self.process or self.process.returncode is not None:
                if not self.connected:
                    logger.warning("Not connected, attempting to connect...")
                    if not await self.connect():
                        return False, "Failed to connect to database"
            
            sql = sql.strip()
            if not sql.endswith(';'):
                sql += ';'
            
            start_time = time.time()
            logger.info(f"Executing: {sql[:80]}...")
            
            try:
                # 쿼리 실행 및 마커 출력
                command = f"{sql}\nPROMPT {self._response_marker}\n"
                self.process.stdin.write(command.encode('utf-8'))
                await self.process.stdin.drain()
                
                # 결과 읽기
                output_lines = []
                
                while True:
                    try:
                        line = await asyncio.wait_for(
                            self.process.stdout.readline(),
                            timeout=timeout
                        )
                    except asyncio.TimeoutError:
                        return False, f"Query timeout after {timeout}s"
                    
                    if not line:
                        self.connected = False
                        return False, "Process terminated unexpectedly"
                    
                    decoded = line.decode('utf-8', errors='replace').rstrip('\r\n')
                    
                    # 마커 체크
                    if self._response_marker in decoded:
                        break
                    
                    # 필터링
                    if decoded.startswith('Picked up JAVA_TOOL_OPTIONS'):
                        continue
                    
                    output_lines.append(decoded)
                
                elapsed = time.time() - start_time
                self._query_count += 1
                self._total_query_time += elapsed
                
                logger.info(f"✅ Completed in {elapsed:.3f}s (avg: {self._total_query_time/self._query_count:.3f}s)")
                
                # 결과 정리
                result = '\n'.join(output_lines).strip()
                
                # 에러 체크
                if self._has_error(result):
                    return False, result
                
                return True, result
                
            except Exception as e:
                logger.error(f"Execution error: {e}")
                return False, f"Execution error: {str(e)}"
    
    def _has_error(self, output: str) -> bool:
        """에러 메시지가 있는지 확인합니다."""
        error_patterns = ['ORA-', 'SP2-', 'Error at', 'PLS-']
        return any(pattern in output for pattern in error_patterns)
    
    def is_connected(self) -> bool:
        return self.connected and self.process is not None and self.process.returncode is None
    
    def get_stats(self) -> dict:
        """통계 정보를 반환합니다."""
        uptime = time.time() - self._connection_time if self._connection_time else 0
        avg_time = self._total_query_time / self._query_count if self._query_count > 0 else 0
        return {
            "connected": self.is_connected(),
            "uptime_seconds": uptime,
            "query_count": self._query_count,
            "avg_query_time": avg_time
        }


# 글로벌 세션
session: Optional[PersistentSQLclSession] = None


async def handle_query(request: web.Request) -> web.Response:
    """SQL 쿼리 실행 핸들러"""
    global session
    
    try:
        data = await request.json()
        sql = data.get('sql', '')
        timeout = data.get('timeout', 60)
        
        if not sql:
            return web.json_response({
                "success": False,
                "error": "SQL query is required"
            }, status=400)
        
        if not session or not session.is_connected():
            return web.json_response({
                "success": False,
                "error": "Database not connected"
            }, status=503)
        
        start = time.time()
        success, result = await session.execute(sql, timeout)
        elapsed = time.time() - start
        
        return web.json_response({
            "success": success,
            "data": result if success else None,
            "error": result if not success else None,
            "elapsed": elapsed
        })
        
    except Exception as e:
        logger.error(f"Query handler error: {e}")
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)


async def handle_status(request: web.Request) -> web.Response:
    """서버 상태 확인 핸들러"""
    global session
    
    if session:
        stats = session.get_stats()
        db_info = DB_CONNECTION.split('@')[1] if '@' in DB_CONNECTION else 'Unknown'
        stats["database"] = db_info
        return web.json_response(stats)
    else:
        return web.json_response({
            "connected": False,
            "error": "Session not initialized"
        })


async def handle_reconnect(request: web.Request) -> web.Response:
    """재연결 핸들러"""
    global session
    
    try:
        if session:
            await session.disconnect()
        
        session = PersistentSQLclSession(SQLCL_PATH, DB_CONNECTION)
        success = await session.connect()
        
        if success:
            return web.json_response({"success": True, "message": "Reconnected"})
        else:
            return web.json_response({"success": False, "error": "Reconnection failed"}, status=503)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def init_app() -> web.Application:
    """앱 초기화"""
    global session
    
    app = web.Application()
    
    # 라우트 등록
    app.router.add_post('/query', handle_query)
    app.router.add_get('/status', handle_status)
    app.router.add_post('/reconnect', handle_reconnect)
    
    # 서버 시작시 DB 연결
    logger.info("=" * 60)
    logger.info("🚀 Starting Persistent DB Server...")
    logger.info("=" * 60)
    
    if DB_CONNECTION:
        session = PersistentSQLclSession(SQLCL_PATH, DB_CONNECTION)
        connected = await session.connect()
        
        if connected:
            logger.info("")
            logger.info("🎉 Server is ready!")
            logger.info(f"   Endpoint: http://{SERVER_HOST}:{SERVER_PORT}")
            logger.info("")
            logger.info("   POST /query   - Execute SQL")
            logger.info("   GET  /status  - Check status")
            logger.info("   POST /reconnect - Reconnect DB")
            logger.info("")
        else:
            logger.error("⚠️ Failed to connect to database!")
            logger.error("   Server will retry on first query")
    else:
        logger.error("❌ DB_CONNECTION not set!")
    
    return app


async def cleanup(app: web.Application):
    """앱 종료시 정리"""
    global session
    if session:
        await session.disconnect()
        logger.info("Server shutdown complete")


def main():
    """메인 함수"""
    app = asyncio.get_event_loop().run_until_complete(init_app())
    app.on_cleanup.append(cleanup)
    
    logger.info(f"Starting HTTP server on {SERVER_HOST}:{SERVER_PORT}...")
    web.run_app(app, host=SERVER_HOST, port=SERVER_PORT, print=None)


if __name__ == "__main__":
    main()
