import subprocess

SQLCL_PATH = r"C:\Users\chiho\sqlcl\bin\sql.exe"
DB_CONNECTION = "KICEIS_DEV/Kiceis_dev1234!@110.45.213.236:11521/ORADB"

sql_query = """
SELECT Z.DEPA_NAME, COUNT(I.EMPL_NUMB) as CNT
FROM INMAST I
JOIN ZME Z ON I.DEPA_CODE = Z.DEPA_CODE
GROUP BY Z.DEPA_NAME
ORDER BY CNT DESC
FETCH FIRST 5 ROWS ONLY;
"""

cmd = [
    SQLCL_PATH,
    "-S",
    DB_CONNECTION,
]

full_sql = f"""
SET PAGESIZE 5000
SET LINESIZE 1000
SET FEEDBACK OFF
SET HEADING ON
{sql_query}
EXIT;
"""

print(f"Executing SQL:\n{sql_query}")

result = subprocess.run(
    cmd,
    input=full_sql,
    capture_output=True,
    text=True,
    encoding='utf-8'
)

print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
