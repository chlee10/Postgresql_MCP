import subprocess
import os
import time
import threading
import queue

# 환경 변수 설정
os.environ["NLS_LANG"] = "KOREAN_KOREA.AL32UTF8"
os.environ["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8"

sqlcl_path = r"C:\Users\chiho\sqlcl\bin\sql.exe"
db_conn = "KICEIS_DEV/Kiceis_dev1234!@110.45.213.236:11521/ORADB"

def reader(pipe, q, name):
    """Reads from a pipe and puts into a queue."""
    try:
        while True:
            char = pipe.read(1)
            if not char:
                break
            q.put((name, char))
            print(char, end='', flush=True)
    except Exception as e:
        print(f"Reader {name} error: {e}")

print(f"Starting SQLcl: {sqlcl_path} ...")

# -S 옵션 제거하여 초기 프롬프트 확인
proc = subprocess.Popen(
    [sqlcl_path, "-S", db_conn],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding='utf-8',
    bufsize=0 # Unbuffered
)

q = queue.Queue()
t_out = threading.Thread(target=reader, args=(proc.stdout, q, "STDOUT"))
t_out.daemon = True
t_out.start()

t_err = threading.Thread(target=reader, args=(proc.stderr, q, "STDERR"))
t_err.daemon = True
t_err.start()

print("Process started. Sending Handshake...")

# Handshake
try:
    # 엔터를 몇 번 쳐서 프롬프트를 유도하거나 반응을 봄
    proc.stdin.write("\nPROMPT READY\n")
    proc.stdin.flush()
    
    # Wait for READY
    buffer = ""
    start = time.time()
    ready = False
    while time.time() - start < 10:
        if not q.empty():
            name, char = q.get()
            buffer += char
            if "READY" in buffer:
                print("\n[SUCCESS] Handshake complete!")
                ready = True
                break
        else:
            time.sleep(0.1)
            
    if not ready:
        print("\n[FAIL] Handshake timed out.")
        print(f"Buffer content: {buffer}")

    if ready:
        print("\n--- Sending Query ---")
        cmd = "SET SQLFORMAT csv\nSELECT 1 FROM DUAL;\nPROMPT ___END___\n"
        proc.stdin.write(cmd)
        proc.stdin.flush()
        
        # Wait for END
        buffer = ""
        start = time.time()
        while time.time() - start < 10:
            if not q.empty():
                name, char = q.get()
                buffer += char
                if "___END___" in buffer:
                    print("\n[SUCCESS] Query complete!")
                    break
            else:
                time.sleep(0.1)

except Exception as e:
    print(f"Error: {e}")

print("\nDone.")
if proc.poll() is None:
    proc.terminate()
