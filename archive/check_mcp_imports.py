try:
    import mcp
    print(f"mcp version: {mcp.__version__}")
    print(f"mcp file: {mcp.__file__}")
    
    try:
        from mcp import ClientSession
        print("ClientSession found in mcp")
    except ImportError:
        print("ClientSession NOT found in mcp")

    try:
        from mcp import StdioServerParameters
        print("StdioServerParameters found in mcp")
    except ImportError:
        print("StdioServerParameters NOT found in mcp")

    try:
        from mcp.client.session import ClientSession
        print("ClientSession found in mcp.client.session")
    except ImportError:
        print("ClientSession NOT found in mcp.client.session")

    try:
        from mcp.client.stdio import StdioServerParameters
        print("StdioServerParameters found in mcp.client.stdio")
    except ImportError:
        print("StdioServerParameters NOT found in mcp.client.stdio")

except ImportError as e:
    print(f"Error importing mcp: {e}")
