"""부서 테이블 확인"""
import asyncio
import sys
sys.path.insert(0, '.')
from sqlcl_mcp.server import SQLclSession
from config import SQLCL_PATH, DB_CONNECTION

async def check():
    session = SQLclSession(SQLCL_PATH, DB_CONNECTION)
    if await session.start():
        # HRM_DEPT 샘플
        result = await session.execute("""
            SELECT * FROM HRM_DEPT WHERE ROWNUM <= 5
        """)
        print('HRM_DEPT sample:')
        print(result)
        print('---')
        
        # INMAST DEPA_CODE 샘플
        result2 = await session.execute("""
            SELECT DISTINCT DEPA_CODE FROM INMAST WHERE ROWNUM <= 10
        """)
        print('INMAST DEPA_CODE sample:')
        print(result2)
        
        await session.close()

asyncio.run(check())
