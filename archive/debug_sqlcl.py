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

print("Process started. Waiting 5 seconds for initialization...")
time.sleep(5)

print("\n--- Sending Query with CSV Format ---")
try:
    # Send setup and query
    cmd = "SET SQLPROMPT ''\nSET SQLFORMAT csv\nSELECT 'TEST_SUCCESS' FROM DUAL;\nPROMPT ___END___\n"
    proc.stdin.write(cmd)
    proc.stdin.flush()
    print(f"Sent: {cmd.strip()}")
    
    time.sleep(2)
    
    # Send exit
    print("\n--- Sending Exit ---")
    proc.stdin.write("EXIT;\n")
    proc.stdin.flush()
    
    time.sleep(1)
except Exception as e:
    print(f"Error writing to stdin: {e}")

print("\nDone.")
if proc.poll() is None:
    proc.terminate()
