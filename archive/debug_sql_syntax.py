import subprocess

SQLCL_PATH = r"C:\Users\chiho\sqlcl\bin\sql.exe"
DB_CONNECTION = "KICEIS_DEV/Kiceis_dev1234!@110.45.213.236:11521/ORADB"

sql_query = """
SELECT * FROM (
    SELECT Z.DEPA_NAME, COUNT(I.EMPL_NUMB) as CNT
    FROM INMAST I
    JOIN ZME Z ON I.DEPA_CODE = Z.DEPA_CODE
    GROUP BY Z.DEPA_NAME
    ORDER BY CNT DESC
) WHERE ROWNUM <= 5;
"""

# Try with ROWNUM first as it is more compatible
print("Testing ROWNUM query...")
cmd = [SQLCL_PATH, "-S", DB_CONNECTION]
full_sql = f"""
SET SQLFORMAT csv
SET FEEDBACK OFF
{sql_query}
EXIT;
"""
result = subprocess.run(cmd, input=full_sql, capture_output=True, text=True, encoding='utf-8')
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)

# Try with FETCH FIRST
print("\nTesting FETCH FIRST query...")
sql_query_fetch = """
SELECT Z.DEPA_NAME, COUNT(I.EMPL_NUMB) as CNT
FROM INMAST I
JOIN ZME Z ON I.DEPA_CODE = Z.DEPA_CODE
GROUP BY Z.DEPA_NAME
ORDER BY CNT DESC
FETCH FIRST 5 ROWS ONLY;
"""
full_sql_fetch = f"""
SET SQLFORMAT csv
SET FEEDBACK OFF
{sql_query_fetch}
EXIT;
"""
result_fetch = subprocess.run(cmd, input=full_sql_fetch, capture_output=True, text=True, encoding='utf-8')
print("STDOUT:", result_fetch.stdout)
print("STDERR:", result_fetch.stderr)
