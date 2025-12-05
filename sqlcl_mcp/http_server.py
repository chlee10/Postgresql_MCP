"""
SQLcl HTTP Server - 상주 서버 (Persistent SQLcl Session)

DB 연결을 유지하며 HTTP API로 SQL 실행
매번 로그인하지 않고 세션 재사용

사용법:
    poetry run python -m sqlcl_mcp.http_server
"""

import asyncio
import logging
import os
import sys
import json
import time
from aiohttp import web

# Config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SQLCL_PATH, DB_CONNECTION, SERVER_HOST, SERVER_PORT, LOG_LEVEL
from sqlcl_mcp.server import SQLclSession

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("sqlcl-http-server")


# =============================================================================
# Global State
# =============================================================================
sql_session: SQLclSession = None
query_count = 0
start_time = time.time()


# =============================================================================
# HTTP Handlers
# =============================================================================
async def handle_execute(request: web.Request) -> web.Response:
    """SQL 실행 핸들러"""
    global sql_session, query_count
    
    try:
        data = await request.json()
        sql = data.get("sql", "")
        timeout = data.get("timeout", 60)
        
        if not sql:
            return web.json_response({"success": False, "error": "SQL required"}, status=400)
        
        if not sql_session or not sql_session.connected:
            return web.json_response({"success": False, "error": "DB not connected"}, status=503)
        
        start = time.time()
        result = await sql_session.execute(sql, timeout)
        elapsed = time.time() - start
        query_count += 1
        
        # 에러 체크
        if result.startswith("ERROR:") or "ORA-" in result or "SP2-" in result:
            return web.json_response({
                "success": False,
                "error": result,
                "elapsed": round(elapsed, 3)
            })
        
        return web.json_response({
            "success": True,
            "data": result,
            "elapsed": round(elapsed, 3)
        })
        
    except json.JSONDecodeError:
        return web.json_response({"success": False, "error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.exception("Execute error")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_status(request: web.Request) -> web.Response:
    """상태 확인"""
    global sql_session, query_count, start_time
    
    db_info = DB_CONNECTION.split('@')[1] if '@' in DB_CONNECTION else 'Unknown'
    
    return web.json_response({
        "connected": sql_session.connected if sql_session else False,
        "database": db_info,
        "uptime_seconds": round(time.time() - start_time, 1),
        "query_count": query_count
    })


async def handle_health(request: web.Request) -> web.Response:
    """헬스 체크"""
    return web.json_response({"status": "ok"})


async def handle_reconnect(request: web.Request) -> web.Response:
    """DB 재연결"""
    global sql_session
    
    if sql_session:
        await sql_session.close()
    
    sql_session = SQLclSession(SQLCL_PATH, DB_CONNECTION)
    connected = await sql_session.start()
    
    return web.json_response({
        "success": connected,
        "message": "Reconnected" if connected else "Reconnect failed"
    })


# =============================================================================
# App Lifecycle
# =============================================================================
async def init_app(app: web.Application):
    """서버 시작시 DB 연결"""
    global sql_session
    
    logger.info("=" * 60)
    logger.info("SQLcl HTTP Server Starting...")
    logger.info("=" * 60)
    logger.info(f"SQLcl Path: {SQLCL_PATH}")
    logger.info(f"DB Connection: {'Set' if DB_CONNECTION else 'NOT SET!'}")
    logger.info(f"Server: http://{SERVER_HOST}:{SERVER_PORT}")
    
    if DB_CONNECTION:
        sql_session = SQLclSession(SQLCL_PATH, DB_CONNECTION)
        connected = await sql_session.start()
        
        if connected:
            logger.info("=" * 60)
            logger.info("✅ SERVER READY - DB Connected!")
            logger.info("=" * 60)
        else:
            logger.error("❌ DB connection failed!")
    else:
        logger.error("❌ DB_CONNECTION not set in .env!")


async def cleanup_app(app: web.Application):
    """서버 종료시 DB 연결 해제"""
    global sql_session
    
    if sql_session:
        await sql_session.close()
    logger.info("Server shutdown complete")


def create_app() -> web.Application:
    """앱 생성"""
    app = web.Application()
    
    app.router.add_post("/execute", handle_execute)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/health", handle_health)
    app.router.add_post("/reconnect", handle_reconnect)
    
    app.on_startup.append(init_app)
    app.on_cleanup.append(cleanup_app)
    
    return app


# =============================================================================
# Main
# =============================================================================
def main():
    print(f"""
╔══════════════════════════════════════════════════════════╗
║         SQLcl HTTP Server (Persistent Session)           ║
╠══════════════════════════════════════════════════════════╣
║  Server: http://{SERVER_HOST}:{SERVER_PORT:<5}                             ║
║                                                          ║
║  Endpoints:                                              ║
║    POST /execute    - Execute SQL query                  ║
║    GET  /status     - Check connection status            ║
║    GET  /health     - Health check                       ║
║    POST /reconnect  - Reconnect to DB                    ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    app = create_app()
    web.run_app(app, host=SERVER_HOST, port=SERVER_PORT, print=None)


if __name__ == "__main__":
    main()
