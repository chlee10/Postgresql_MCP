# SQLcl MCP Server

Oracle SQLcl을 MCP(Model Context Protocol) 서버로 연동하여  
Claude Desktop, VS Code, Streamlit UI에서 자연어로 데이터베이스를 쿼리할 수 있습니다.

---

## 🚀 주요 기능

| 기능 | 설명 |
|------|------|
| **MCP SSE 서버** | HTTP SSE 방식 상주 서버 (기본 모드, DB 연결 유지) |
| **MCP stdio 서버** | Claude Desktop 연동용 (`--stdio` 옵션) |
| **영속 세션** | SQLcl 프로세스 유지로 빠른 응답 |
| **Streamlit UI** | 웹 기반 자연어 SQL 클라이언트 |
| **AI SQL 생성** | Claude/GPT로 자연어 → SQL 변환 |
| **Plotly 차트** | 파이/막대/라인/영역 차트 (데이터 순서 유지) |
| **대화형 컨텍스트** | 이전 쿼리를 기억하여 후속 질문 지원 |

---

## 📦 설치

```bash
# Poetry 사용 (권장)
poetry install

# 또는 pip
pip install -e .
```

---

## ⚙️ 환경 설정

### `.env` 파일 생성

프로젝트 루트에 `.env` 파일을 만들고 민감 정보를 설정합니다.  
(모든 설정의 기본값은 `config.py`에 정의되어 있습니다)

```env
# =============================================================================
# Oracle Database 연결 (필수)
# =============================================================================
# 방법 1: 전체 연결 문자열
DB_CONNECTION=scott/tiger@localhost:1521/ORCL

# 방법 2: 개별 설정 (DB_CONNECTION이 비어있을 때 사용)
# DB_HOST=localhost
# DB_PORT=1521
# DB_SERVICE=ORCL
# DB_USER=scott
# DB_PASSWORD=tiger

# =============================================================================
# AI API 키 (필수)
# =============================================================================
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx

# =============================================================================
# 선택 설정 (기본값은 config.py 참조)
# =============================================================================
# SQLCL_PATH=C:\path\to\sqlcl\bin\sql.exe
# MCP_SERVER_HOST=127.0.0.1
# MCP_SERVER_PORT=8765
# DEFAULT_AI_MODEL=claude-haiku-4-5-20251001
# SQLCL_TIMEOUT=60
# LOG_LEVEL=INFO
```

---

## 🖥️ 실행 방법

### SSE 모드 (Streamlit UI 사용 - 기본)

SSE 모드는 상주 서버로 DB 연결을 유지하여 빠른 응답을 제공합니다.

```bash
# 1. MCP SSE 서버 시작 (터미널 1)
poetry run python -m sqlcl_mcp.server

# 2. Streamlit 앱 시작 (터미널 2)
poetry run streamlit run streamlit_app.py
```

브라우저에서 `http://localhost:8501` 접속

### stdio 모드 (Claude Desktop 연동 시)

```bash
poetry run python -m sqlcl_mcp.server --stdio
```

---

## 🔗 Claude Desktop 연동

`%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sqlcl": {
      "command": "poetry",
      "args": ["run", "python", "-m", "sqlcl_mcp.server", "--stdio"],
      "cwd": "C:\\path\\to\\SQLcl_MCP"
    }
  }
}
```

---

## 🛠️ MCP 도구

| 도구 | 설명 |
|------|------|
| `execute_sql` | SQL 쿼리 실행 (CSV 형식 반환) |
| `get_tables` | 테이블 목록 조회 |
| `describe_table` | 테이블 구조 조회 |
| `get_status` | 연결 상태 확인 |

---

## 💬 Streamlit UI 사용법

### 자연어 쿼리 예시

| 쿼리 | 설명 |
|------|------|
| `부서별 인원수 Top 10을 원그래프로 그려줘` | 파이 차트 |
| `직급별 인원 분포를 막대그래프로 보여줘` | 막대 차트 |
| `최근 10년간 연도별 입사자 수 추이를 라인차트로` | 라인 차트 |
| `연령대별 인원 비율을 원그래프로` | 연령 분석 |
| `근속연수 구간별 인원수 막대그래프` | 근속 분석 |
| `부서별 인원수와 평균 근속연수를 함께 보여줘` | 복합 분석 |
| `40대 이상 직원의 부서별 현황` | 연령 필터 |
| `홍길동 직원 정보` | 개인 상세 조회 |

### 차트 키워드

| 키워드 | 차트 타입 |
|--------|----------|
| 원그래프, 파이, 비율 | 🥧 파이 차트 |
| 막대, 바 | 📊 막대 차트 |
| 라인, 추이, 추세 | 📈 라인 차트 |
| 영역, area | 📉 영역 차트 |

### 대화형 컨텍스트

이전 쿼리 결과를 기억하여 후속 질문이 가능합니다:
```
사용자: 부서별 인원수 Top 5 보여줘
AI: (결과 표시)
사용자: 이걸 원그래프로 그려줘
AI: (이전 결과를 파이 차트로 표시)
```

---

## 📁 프로젝트 구조

```
SQLcl_MCP/
├── config.py              # 모든 설정값 (환경 변수 + 기본값)
├── streamlit_app.py       # Streamlit UI (SSE 클라이언트)
├── pyproject.toml         # Poetry 설정
├── requirements.txt       # pip 의존성
├── .env                   # 민감 정보 (Git 무시됨)
├── .gitignore             # Git 제외 파일 목록
├── README.md              # 사용 가이드
├── ARCHITECTURE.md        # 기술 구조 및 사양 상세
├── query.md               # 쿼리 예시 모음
├── sqlcl_mcp/
│   ├── __init__.py
│   └── server.py          # MCP 서버 (SSE 기본, --stdio 옵션)
└── old/                   # 아카이브
```

---

## 🔐 보안

| 항목 | 위치 | 설명 |
|------|------|------|
| `DB_USER` | `.env` | 데이터베이스 사용자 |
| `DB_PASSWORD` | `.env` | 데이터베이스 비밀번호 |
| `DB_CONNECTION` | `.env` | 전체 연결 문자열 |
| `ANTHROPIC_API_KEY` | `.env` | Claude API 키 |
| `OPENAI_API_KEY` | `.env` | OpenAI API 키 |

> ⚠️ `.env` 파일은 `.gitignore`에 포함되어 Git에 커밋되지 않습니다.  
> 민감 정보는 절대 `config.py`에 하드코딩하지 마세요.

---

## 🔧 아키텍처

```
┌─────────────────┐     SSE/HTTP      ┌─────────────────┐
│  Streamlit UI   │ ◄──────────────► │  MCP SSE Server │
│  (Port 8501)    │                   │  (Port 8765)    │
└─────────────────┘                   └────────┬────────┘
        │                                      │
        │ Claude/GPT API                       │
        ▼                                      ▼
┌─────────────────┐                   ┌─────────────────┐
│  AI SQL 생성    │                   │  SQLcl Session  │
│  (자연어→SQL)   │                   │  (Persistent)   │
└─────────────────┘                   └────────┬────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │  Oracle DB      │
                                      └─────────────────┘
```

---

## 🛡️ 기술 스택

- **Python 3.10+**
- **SQLcl** - Oracle SQL Command Line
- **MCP** - Model Context Protocol (SSE 기본 + stdio 옵션)
- **Starlette + Uvicorn** - SSE 서버
- **Streamlit** - 웹 UI
- **Plotly** - 인터랙티브 차트
- **Anthropic Claude / OpenAI GPT** - AI SQL 생성

---

## 📌 버전 히스토리

| 버전 | 날짜 | 주요 변경 |
|------|------|----------|
| v2.1.0 | 2025-12-03 | SSE 기본 모드, Plotly 차트, 대화 컨텍스트, 민감정보 .env 분리 |
| v2.0.0 | 2025-12 | MCP SSE 서버, Streamlit UI 추가 |
| v1.0.0 | 2025-11 | 초기 stdio MCP 서버 |

---

## 📚 추가 문서

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - 기술 구조, 데이터 흐름, SQL 생성 규칙 상세
- **[query.md](query.md)** - 자연어 쿼리 예시 모음

---

## 📄 라이선스

MIT License

