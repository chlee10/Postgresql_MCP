import mcp
print(f"mcp package: {mcp}")

try:
    from mcp import ClientSession
    print("SUCCESS: from mcp import ClientSession")
except ImportError as e:
    print(f"FAIL: from mcp import ClientSession ({e})")

try:
    from mcp import StdioServerParameters
    print("SUCCESS: from mcp import StdioServerParameters")
except ImportError as e:
    print(f"FAIL: from mcp import StdioServerParameters ({e})")

try:
    from mcp.client.session import ClientSession
    print("SUCCESS: from mcp.client.session import ClientSession")
except ImportError as e:
    print(f"FAIL: from mcp.client.session import ClientSession ({e})")

try:
    from mcp.client.stdio import StdioServerParameters
    print("SUCCESS: from mcp.client.stdio import StdioServerParameters")
except ImportError as e:
    print(f"FAIL: from mcp.client.stdio import StdioServerParameters ({e})")
