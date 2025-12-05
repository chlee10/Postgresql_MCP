import asyncio
import sys
sys.path.insert(0, '.')
from sqlcl_mcp.server import SQLclSession
from config import SQLCL_PATH, DB_CONNECTION

# Test alias extraction
from streamlit_app import extract_column_aliases, parse_csv_with_headers

# Test SQL queries
test_sqls = [
    "SELECT Z.DEPA_NAME AS DEPT_NAME, COUNT(*) AS CNT FROM INMAST M JOIN ZME Z ON M.DEPA_CODE = Z.DEPA_CODE GROUP BY Z.DEPA_NAME",
    "SELECT table_name AS TABLE_NAME FROM user_tables",
    "SELECT COUNT(*) AS TOTAL, SUM(CASE WHEN SEX_GUBN = '1' THEN 1 ELSE 0 END) AS MALE FROM INMAST"
]

print("=== Testing extract_column_aliases ===")
for sql in test_sqls:
    aliases = extract_column_aliases(sql)
    print(f"SQL: {sql[:60]}...")
    print(f"Aliases: {aliases}")
    print()

async def check():
    session = SQLclSession(SQLCL_PATH, DB_CONNECTION)
    if await session.start():
        sql = 'SELECT Z.DEPA_NAME AS DEPT_NAME, COUNT(*) AS CNT FROM INMAST M JOIN ZME Z ON M.DEPA_CODE = Z.DEPA_CODE GROUP BY Z.DEPA_NAME FETCH FIRST 3 ROWS ONLY'
        result = await session.execute(sql)
        print('=== SQLcl Raw output ===')
        print(repr(result))
        print()
        
        print('=== Parsed DataFrame ===')
        df = parse_csv_with_headers(result, sql)
        print(df)
        print()
        print('Columns:', list(df.columns))
        
        await session.close()

asyncio.run(check())
