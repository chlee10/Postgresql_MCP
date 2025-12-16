"""
SQLcl HTTP Server - Oracle SQLcl 통합 HTTP API

파일 기반 실행으로 안정적인 SQLcl 연동
HTTP API로 Streamlit 앱에서 접근

사용법:
    python -m sqlcl_mcp.http_server
"""

import asyncio
import logging
import os
import sys
import json
import time
import tempfile
import subprocess
from typing import Optional
from aiohttp import web
from concurrent.futures import ThreadPoolExecutor

# Config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SQLCL_PATH, DB_CONNECTION, SERVER_HOST, SERVER_PORT,
    SQLCL_INIT_COMMANDS, LOG_LEVEL, LOG_FORMAT
)

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger("sqlcl-http-server")

# Thread pool for blocking operations
thread_executor = ThreadPoolExecutor(max_workers=4)


# =============================================================================
# SQLcl Executor
# =============================================================================
class SQLclExecutor:
    """파일 기반 SQLcl 실행기 - 안정적인 실행 보장"""
    
    def __init__(self, sqlcl_path: str, db_connection: str):
        self.sqlcl_path = sqlcl_path
        self.db_connection = db_connection
        self._query_count = 0
        self._start_time = time.time()
        self._connected = False
        self._lock = asyncio.Lock()
    
    def _get_env(self) -> dict:
        """SQLcl 실행 환경 변수"""
        env = os.environ.copy()
        env["NLS_LANG"] = "KOREAN_KOREA.AL32UTF8"
        env["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8"
        return env
    
    def _run_query_sync(self, sql: str, timeout: float = 60.0) -> tuple[bool, str]:
        """동기 방식 쿼리 실행 (ThreadPool에서 실행)"""
        sql = sql.strip()
        if not sql.endswith(';'):
            sql += ';'
        
        sql_content = f"{SQLCL_INIT_COMMANDS}\n{sql}\n\nEXIT;\n"
        sql_file = None
        
        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False, encoding='utf-8') as f:
                f.write(sql_content)
                sql_file = f.name
            
            # SQLcl 실행
            result = subprocess.run(
                [self.sqlcl_path, "-S", self.db_connection, f"@{sql_file}"],
                capture_output=True, text=True, timeout=timeout,
                encoding='utf-8', errors='replace', env=self._get_env()
            )
            
            # 출력 필터링
            stdout = self._filter_output(result.stdout)
            stderr = self._filter_output(result.stderr)
            
            # 에러 체크
            if 'ORA-' in stdout or 'SP2-' in stdout:
                return False, stdout
            
            if result.returncode != 0 and not stdout:
                return False, stderr or f"Exit code: {result.returncode}"
            
            return True, stdout
            
        except subprocess.TimeoutExpired:
            return False, f"Query timeout ({timeout}s)"
        except FileNotFoundError:
            return False, f"SQLcl not found: {self.sqlcl_path}"
        except Exception as e:
            return False, f"Error: {str(e)}"
        finally:
            if sql_file and os.path.exists(sql_file):
                try:
                    os.remove(sql_file)
                except Exception:
                    pass
    
    def _filter_output(self, text: str) -> str:
        """JAVA_TOOL_OPTIONS 메시지 필터링"""
        if not text:
            return ""
        lines = [line for line in text.split('\n') if not line.startswith('Picked up JAVA_TOOL_OPTIONS')]
        return '\n'.join(lines).strip()
    
    async def execute(self, sql: str, timeout: float = 60.0) -> tuple[bool, str, float]:
        """비동기 쿼리 실행"""
        async with self._lock:
            start = time.time()
            
            loop = asyncio.get_event_loop()
            success, result = await loop.run_in_executor(
                thread_executor, self._run_query_sync, sql, timeout
            )
            
            elapsed = time.time() - start
            self._query_count += 1
            
            if success:
                self._connected = True
            
            logger.info(f"Query #{self._query_count} completed in {elapsed:.3f}s (success={success})")
            return success, result, elapsed
    
    async def test_connection(self) -> bool:
        """연결 테스트"""
        logger.info("Testing database connection...")
        success, _, elapsed = await self.execute("SELECT 'OK' FROM DUAL", timeout=30)
        self._connected = success
        logger.info(f"Connection test: {'✅ OK' if success else '❌ Failed'} ({elapsed:.2f}s)")
        return success
    
    def is_connected(self) -> bool:
        return self._connected
    
    def get_stats(self) -> dict:
        return {
            "connected": self._connected,
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "query_count": self._query_count
        }


# =============================================================================
# HTTP Handlers
# =============================================================================
sql_executor: Optional[SQLclExecutor] = None


async def handle_execute(request: web.Request) -> web.Response:
    """SQL 실행 핸들러"""
    global sql_executor
    
    try:
        data = await request.json()
        sql = data.get("sql", "")
        timeout = data.get("timeout", 60)
        
        if not sql:
            return web.json_response({"success": False, "error": "SQL required"}, status=400)
        
        if not sql_executor:
            return web.json_response({"success": False, "error": "Server not initialized"}, status=503)
        
        success, result, elapsed = await sql_executor.execute(sql, timeout)
        
        return web.json_response({
            "success": success,
            "data": result if success else None,
            "error": result if not success else None,
            "elapsed": round(elapsed, 3)
        })
        
    except json.JSONDecodeError:
        return web.json_response({"success": False, "error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.exception("Execute error")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_status(request: web.Request) -> web.Response:
    """상태 확인"""
    if sql_executor:
        stats = sql_executor.get_stats()
        stats["database"] = DB_CONNECTION.split('@')[1] if '@' in DB_CONNECTION else 'Unknown'
        return web.json_response(stats)
    return web.json_response({"connected": False, "error": "Not initialized"})


async def handle_health(request: web.Request) -> web.Response:
    """헬스 체크"""
    return web.json_response({"status": "ok"})


# =============================================================================
# App Lifecycle
# =============================================================================
async def init_app(app: web.Application):
    """서버 시작시 초기화"""
    global sql_executor
    
    logger.info("=" * 60)
    logger.info("SQLcl HTTP Server Starting...")
    logger.info("=" * 60)
    logger.info(f"SQLcl Path: {SQLCL_PATH}")
    logger.info(f"DB Connection: {'Set' if DB_CONNECTION else 'NOT SET!'}")
    logger.info(f"Server: http://{SERVER_HOST}:{SERVER_PORT}")
    
    if DB_CONNECTION:
        sql_executor = SQLclExecutor(SQLCL_PATH, DB_CONNECTION)
        connected = await sql_executor.test_connection()
        
        if connected:
            logger.info("=" * 60)
            logger.info("✅ SERVER READY!")
            logger.info("=" * 60)
        else:
            logger.warning("⚠️ Initial connection failed - will retry on requests")
    else:
        logger.error("❌ DB_CONNECTION not set!")


async def cleanup_app(app: web.Application):
    """서버 종료시 정리"""
    thread_executor.shutdown(wait=False)
    logger.info("Server shutdown complete")


def create_app() -> web.Application:
    """앱 생성"""
    app = web.Application()
    
    app.router.add_post("/execute", handle_execute)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/health", handle_health)
    
    app.on_startup.append(init_app)
    app.on_cleanup.append(cleanup_app)
    
    return app


# =============================================================================
# Main
# =============================================================================
def main():
    print(f"""
╔══════════════════════════════════════════════════════════╗
║              SQLcl HTTP Server                           ║
╠══════════════════════════════════════════════════════════╣
║  Server: http://{SERVER_HOST}:{SERVER_PORT:<5}                             ║
║                                                          ║
║  Endpoints:                                              ║
║    POST /execute  - Execute SQL query                    ║
║    GET  /status   - Check connection status              ║
║    GET  /health   - Health check                         ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    app = create_app()
    web.run_app(app, host=SERVER_HOST, port=SERVER_PORT, print=None)


if __name__ == "__main__":
    main()
