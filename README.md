# SQLcl MCP Server

Oracle SQLcl을 MCP(Model Context Protocol) 서버로 연동하여  
Claude Desktop, VS Code 등에서 자연어로 데이터베이스를 쿼리할 수 있습니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **MCP 서버** | stdio 방식 MCP 프로토콜 지원 |
| **영속 세션** | SQLcl 프로세스 유지로 빠른 응답 |
| **Fallback** | 세션 실패 시 파일 기반 실행 |
| **Streamlit UI** | 웹 기반 자연어 SQL 클라이언트 (별도) |

---

## 빠른 시작

### 1. 설치

```bash
# Poetry 사용
poetry install

# 또는 pip
pip install -e .
```

### 2. 환경 변수 설정

`.env` 파일 생성:

```env
# Oracle SQLcl 경로
SQLCL_PATH=C:\path\to\sqlcl\bin\sql.exe

# 데이터베이스 연결 정보
DB_CONNECTION=username/password@host:port/service

# AI API 키 (Streamlit 앱 사용 시)
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx
```

### 3. MCP 서버 실행

```bash
poetry run python -m sqlcl_mcp.server
```

### 4. Claude Desktop 연동

`%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sqlcl": {
      "command": "poetry",
      "args": ["run", "python", "-m", "sqlcl_mcp.server"],
      "cwd": "C:\\path\\to\\SQLcl_MCP"
    }
  }
}
```

---

## MCP 도구

| 도구 | 설명 |
|------|------|
| `execute_sql` | SQL 쿼리 실행 (CSV 형식 반환) |
| `get_tables` | 테이블 목록 조회 |
| `describe_table` | 테이블 구조 조회 |
| `get_status` | 연결 상태 확인 |

---

## Streamlit UI (별도)

자연어로 SQL을 생성하고 실행하는 웹 UI:

```bash
poetry run streamlit run streamlit_app.py
```

### 기능
- 자연어 → SQL 변환 (Claude/GPT)
- 결과 테이블 + 차트 시각화
- 대화형 질의

### 차트 키워드

| 키워드 | 차트 |
|--------|------|
| 원그래프, 파이, 비율 | 파이 차트 |
| 막대, 바 | 막대 차트 |
| 라인, 추이, 추세 | 라인 차트 |

---

## 프로젝트 구조

```
SQLcl_MCP/
├── config.py              # 설정 상수
├── streamlit_app.py       # Streamlit UI
├── pyproject.toml         # Poetry 설정
├── .env                   # 환경 변수
├── sqlcl_mcp/
│   ├── __init__.py
│   └── server.py          # MCP stdio 서버
└── old/                   # 아카이브
```

---

## 기술 스택

- **Python 3.10+**
- **SQLcl** - Oracle CLI
- **MCP** - Model Context Protocol
- **Streamlit** - 웹 UI
- **Anthropic/OpenAI** - LLM API

---

## 라이선스

MIT License
