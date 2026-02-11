# PostgreSQL MCP AI Explorer

> **v1.2.0** | MCP SDK 1.26 · psycopg 3 · Streamlit

PostgreSQL 데이터베이스를 **MCP(Model Context Protocol)** 로 연결하여,  
AI와 대화하며 자연어로 데이터를 탐색하고 시각화하는 프로젝트입니다.

---

## 🚀 주요 기능

| 기능 | 설명 |
|------|------|
| **자연어 → SQL 변환** | 복잡한 SQL 문법 없이 질문으로 데이터 조회 |
| **MCP 프로토콜 지원** | Claude Desktop(stdio) 및 기타 MCP 클라이언트(SSE)와 연동 |
| **Streamlit Web UI** | 채팅 인터페이스 + Plotly 차트 자동 생성 |
| **실전형 데모 데이터** | 한국형 인사관리 데이터셋 자동 생성 스크립트 포함 |

---

## 🛠️ 설치 및 설정

### 1. 필수 요구사항
- Python 3.10 이상
- PostgreSQL 데이터베이스 (Local 또는 Docker)
- [Poetry](https://python-poetry.org/) (패키지 관리자)

### 2. 설치
```bash
# 의존성 설치
poetry install
```

### 3. 환경 설정 (.env)
프로젝트 루트에 `.env` 파일을 생성하고 데이터베이스 정보를 입력하세요.
```ini
# PostgreSQL 접속 정보
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password

# (선택) AI API 키 — Streamlit 웹 UI에서 자연어→SQL 변환 시 필요
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

---

## 🏃‍♂️ 실행 방법

### 1. 데이터베이스 초기화 (데모 데이터 생성)
실습용 `인사관리` 테이블을 생성하고 약 50명의 가상 사원 정보를 채웁니다.
```bash
poetry run python scripts/setup_demo_data.py
```

### 2-A. Claude Desktop 연동 (stdio 모드, 기본)
Claude Desktop의 설정 파일 `%APPDATA%\Claude\claude_desktop_config.json` 에 아래 내용을 추가합니다.
```jsonc
{
  "mcpServers": {
    "postgresql": {
      "command": "<프로젝트경로>\\.venv\\Scripts\\python.exe",
      "args": ["-m", "src.server"],
      "env": {
        "PYTHONPATH": "<프로젝트경로>"
      }
    }
  }
}
```
설정 후 Claude Desktop을 **재시작**하면 자동으로 MCP 서버가 연결됩니다.

### 2-B. Streamlit 웹 UI (SSE 모드)
SSE 백엔드 서버와 Streamlit 프론트엔드를 각각 실행합니다.
```bash
# 터미널 1: MCP 서버 (SSE 모드)
poetry run python -m src.server --sse

# 터미널 2: Streamlit 앱
poetry run streamlit run src/streamlit_app.py
```

---

## 🧪 테스트 쿼리 예시

- "개발팀의 평균 연봉은 얼마인가요?"
- "근속 3년 이상인 직원 중 급여 상위 5명 보여줘"
- "부서별 남녀 성비를 파이차트로 그려줘"
- "부서별 인원수 Top 5과 평균 근속연수를 함께 보여줘"
- "올해 입사한 신입사원 명단"

> 더 많은 예시는 [query.md](query.md) 를 참고하세요.

---

## 📁 프로젝트 구조

```text
PostgreSQL_MCP/
├── src/                       # 핵심 패키지 (서버 + 설정 + 웹 UI)
│   ├── __init__.py            #   패키지 메타 (버전 정보)
│   ├── server.py              #   MCP 서버 핵심 로직 (stdio + SSE 전송)
│   ├── config.py              #   통합 설정 (DB, MCP, AI 프롬프트 등)
│   └── streamlit_app.py       #   Streamlit 웹 UI (채팅 + 차트)
├── scripts/
│   └── setup_demo_data.py     #   데모 데이터(인사관리) 생성 스크립트
├── docs/
│   └── ARCHITECTURE.md        #   시스템 아키텍처 문서
├── query.md                   # 추천 자연어 쿼리 예시 모음
├── pyproject.toml             # Poetry 의존성 관리
├── claude_desktop_config.json # Claude Desktop 설정 예시
└── archive/                   # 이전 버전 / 디버깅용 파일 보관
```
