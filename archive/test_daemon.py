import subprocess
import time
import os

# 환경 변수 설정 (한글 처리)
os.environ["NLS_LANG"] = "KOREAN_KOREA.AL32UTF8"
os.environ["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8"

sqlcl_path = r"C:\Users\chiho\sqlcl\bin\sql.exe"
db_conn = "KICEIS_DEV/Kiceis_dev1234!@110.45.213.236:11521/ORADB"

print("Starting SQLcl process...")
proc = subprocess.Popen(
    [sqlcl_path, "-S", db_conn],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding='utf-8',
    bufsize=0  # Unbuffered
)

def run_query(query):
    print(f"Sending query: {query}")
    # SQLFormat 설정과 쿼리, 그리고 종료 마커 전송
    full_cmd = f"SET SQLFORMAT csv\n{query}\nPROMPT ___END___\n"
    proc.stdin.write(full_cmd)
    proc.stdin.flush()
    
    output = []
    while True:
        line = proc.stdout.readline()
        # print(f"DEBUG: {line.strip()}") # 디버깅용
        if not line:
            break
        
        clean_line = line.strip()
        if clean_line == "___END___":
            break
        if clean_line: # 빈 줄 제외
            output.append(clean_line)
            
    return "\n".join(output)

try:
    # 초기화 대기 (필요시)
    time.sleep(2)
    
    # Test 1
    print("\n--- Test 1 ---")
    res1 = run_query("SELECT 'Connection Alive' as status FROM dual;")
    print(res1)

    # Test 2
    print("\n--- Test 2 ---")
    res2 = run_query("SELECT count(*) as cnt FROM INMAST;")
    print(res2)

finally:
    print("\nClosing process...")
    proc.terminate()
