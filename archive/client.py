import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 설정: 서버 실행 정보 및 환경 변수
SERVER_MODULE = "sqlcl_mcp.server"
SQLCL_PATH = r"C:\Users\chiho\sqlcl\bin\sql.exe"
DB_CONNECTION = "KICEIS_DEV/Kiceis_dev1234!@110.45.213.236:11521/ORADB"

async def run():
    # 1. 서버 파라미터 설정
    # python -m sqlcl_mcp.server 명령어로 서버를 실행합니다.
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", SERVER_MODULE],
        env={
            "SQLCL_PATH": SQLCL_PATH,
            "DB_CONNECTION": DB_CONNECTION,
            **os.environ # 기존 환경 변수 포함
        }
    )

    print("MCP 서버에 연결 중...", end="", flush=True)

    # 2. 서버 연결 및 세션 시작
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 초기화
            await session.initialize()
            print(" 완료!")
            
            # 3. 사용 가능한 도구 목록 가져오기
            tools_result = await session.list_tools()
            tools = tools_result.tools
            
            print(f"\n[연결됨] SQLcl MCP Server")
            print(f"사용 가능한 도구: {len(tools)}개")

            while True:
                print("\n" + "="*30)
                print("도구를 선택하세요 (종료: q):")
                for i, tool in enumerate(tools):
                    print(f"{i+1}. {tool.name}")
                print("="*30)
                
                choice = input("선택 > ").strip()
                if choice.lower() in ['q', 'quit', 'exit']:
                    break
                
                try:
                    idx = int(choice) - 1
                    if idx < 0 or idx >= len(tools):
                        print("잘못된 번호입니다.")
                        continue
                    selected_tool = tools[idx]
                except ValueError:
                    print("숫자를 입력해주세요.")
                    continue

                # 4. 도구 인자 입력 받기
                args = {}
                print(f"\n[{selected_tool.name}] 실행을 위한 정보를 입력하세요:")
                print(f"설명: {selected_tool.description}")
                
                if selected_tool.inputSchema and "properties" in selected_tool.inputSchema:
                    schema = selected_tool.inputSchema
                    required_props = schema.get("required", [])
                    
                    for prop_name, prop_info in schema["properties"].items():
                        # connection은 환경변수로 설정했으므로 입력받지 않고 건너뜀 (옵션인 경우)
                        if prop_name == "connection":
                            continue
                            
                        is_required = prop_name in required_props
                        desc = prop_info.get('description', '')
                        prompt = f"- {prop_name} ({desc})"
                        if is_required:
                            prompt += " [필수]"
                        
                        val = input(f"{prompt}: ")
                        if val:
                            args[prop_name] = val
                        elif is_required:
                            print(f"오류: {prop_name}은(는) 필수 입력 항목입니다.")
                            break
                    else:
                        # break 없이 루프가 끝났다면 (모든 필수 입력 완료) 실행
                        await execute_tool(session, selected_tool.name, args)
                else:
                    # 인자가 없는 경우 바로 실행
                    await execute_tool(session, selected_tool.name, {})

async def execute_tool(session, tool_name, args):
    print(f"\nRunning {tool_name}...")
    try:
        result = await session.call_tool(tool_name, arguments=args)
        
        print("\n--- 실행 결과 ---")
        for content in result.content:
            if content.type == "text":
                print(content.text)
        print("-----------------")
    except Exception as e:
        print(f"\n오류 발생: {e}")

if __name__ == "__main__":
    # 윈도우에서 asyncio 이벤트 루프 정책 설정 (필요한 경우)
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
