import subprocess

SQLCL_PATH = r"C:\Users\chiho\sqlcl\bin\sql.exe"
DB_CONNECTION = "KICEIS_DEV/Kiceis_dev1234!@110.45.213.236:11521/ORADB"

sql_query = "DESC ZME;"

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

result = subprocess.run(
    cmd,
    input=full_sql,
    capture_output=True,
    text=True,
    encoding='utf-8'
)

print(result.stdout)
print(result.stderr)
