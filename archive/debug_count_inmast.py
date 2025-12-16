import os
import sys

# Set environment variable for SQLCL_PATH
os.environ["SQLCL_PATH"] = r"C:\Users\chiho\sqlcl\bin\sql.exe"

try:
    from sqlcl_mcp.server import execute_sqlcl, SQLCL_PATH
    # Patch it just in case
    import sqlcl_mcp.server
    sqlcl_mcp.server.SQLCL_PATH = r"C:\Users\chiho\sqlcl\bin\sql.exe"
    
    print(f"Using SQLCL_PATH: {sqlcl_mcp.server.SQLCL_PATH}")
    
    connection = "KICEIS_DEV/Kiceis_dev1234!@110.45.213.236:11521/ORADB"
    table_name = "INMAST"
    sql = f"SELECT COUNT(*) FROM {table_name};"
    
    print(f"Executing query: {sql}")
    result = execute_sqlcl(sql, connection)
    
    if result["success"]:
        print("\n--- Row Count (INMAST) ---")
        print(result["output"])
        print("--------------------------")
    else:
        print("\nError executing SQL:")
        print(result["error"])
        if result.get("stderr"):
            print("Stderr:", result["stderr"])

except ImportError as e:
    print(f"Failed to import sqlcl_mcp: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
