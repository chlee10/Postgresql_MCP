import subprocess
import os
import tempfile

SQLCL_PATH = r"C:\Users\chiho\sqlcl\bin\sql.exe"
DB_CONNECTION = "KICEIS_DEV/Kiceis_dev1234!@110.45.213.236:11521/ORADB"

sql_query = """
SET SQLFORMAT csv
SET FEEDBACK OFF
SELECT column_name, data_type FROM user_tab_columns WHERE table_name = 'HRM_PERSON' ORDER BY column_name;
EXIT;
"""

try:
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sql', encoding='utf-8') as tmp:
        tmp.write(sql_query)
        tmp_path = tmp.name

    print(f"Created temp file: {tmp_path}")

    cmd = [SQLCL_PATH, "-S", DB_CONNECTION, f"@{tmp_path}"]
    
    env = os.environ.copy()
    env["NLS_LANG"] = "KOREAN_KOREA.AL32UTF8"
    env["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8"

    print("Running SQLcl...")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        env=env
    )

    print("--- STDOUT ---")
    print(result.stdout)
    print("--- STDERR ---")
    print(result.stderr)

    os.remove(tmp_path)

except Exception as e:
    print(f"Error: {e}")
