# PostgreSQL MCP AI Explorer — 기술 구조 및 사양

> 버전: 1.1.0 | 최종 수정: 2026-02-09

---

## 📐 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client Layer                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Streamlit UI   │  │  Claude Desktop │  │  VS Code /      │              │
│  │  (Port 8501)    │  │  (stdio)        │  │  MCP Client     │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
│           │                    │                    │                        │
│           │ SSE/HTTP           │ stdio              │ stdio/SSE             │
│           ▼                    ▼                    ▼                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                              MCP Server Layer                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    MCP Server (server.py)                           │    │
│  │  ┌─────────────────┐              ┌─────────────────┐               │    │
│  │  │  SSE Transport  │              │ stdio Transport │               │    │
│  │  │  (--sse 옵션)   │              │ (기본 모드)     │               │    │
│  │  │  Port: 8765     │              │                 │               │    │
│  │  └─────────────────┘              └─────────────────┘               │    │
│  │                                                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                    MCP Tools                                 │    │    │
│  │  │  • query           - SQL 쿼리 실행 (결과 JSON)               │    │    │
│  │  │  • list_tables     - 테이블 목록 조회                         │    │    │
│  │  │  • describe_table  - 테이블 컬럼 구조 조회                    │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│                              Database Layer                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              PostgresManager (psycopg 3 + AsyncConnectionPool)     │    │
│  │  • 비동기 커넥션 풀 (min_size=1, max_size=10)                       │    │
│  │  • dict_row 팩토리 — 결과를 Dict[str, Any] 리스트로 반환            │    │
│  └────────────────────────────┬────────────────────────────────────────┘    │
│                               │                                              │
│                               │ TCP (libpq)                                  │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         PostgreSQL Database                          │    │
│  │  • Connection: postgresql://user:password@host:port/dbname           │    │
│  │  • 주요 테이블: 인사관리 (한국형 인사 데모 데이터)                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 데이터 흐름

### 1. 자연어 → SQL 변환 흐름 (Streamlit)

```
┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ 사용자    │───►│ Streamlit UI │───►│ AI API      │───►│ SQL 생성     │
│ 자연어    │    │ (입력 처리)  │    │ Claude/GPT  │    │ (정제된 SQL) │
└──────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

### 2. SQL 실행 흐름

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Streamlit UI │───►│ MCP Client   │───►│ MCP Server   │───►│ PostgreSQL   │
│ (SSE 요청)   │    │ (mcp SDK)    │    │ (Starlette)  │    │ (psycopg 3)  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                                                                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 차트/테이블   │◄───│ DataFrame    │◄───│ JSON 파싱    │◄───│ dict_row     │
│ (Plotly)     │    │ (Pandas)     │    │              │    │ (결과 반환)  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### 3. Claude Desktop 직접 연동 (stdio)

```
Claude Desktop ◄──stdin/stdout──► MCP Server ◄──psycopg──► PostgreSQL
                  (JSON-RPC)         (server.py)              (DB)
```

---

## 📦 모듈 구조

### 핵심 파일

| 파일 | 역할 | 주요 클래스/함수 |
|------|------|------------------|
| `postgresql_mcp/server.py` | MCP 서버 | `PostgresManager`, `MCPServer`, `main()` |
| `streamlit_app.py` | 웹 UI | `display_data()`, `display_chart()`, `generate_sql_from_nl()` |
| `config.py` | 통합 설정 | 환경변수, DB 스키마, AI 모델, SQL 생성 규칙 |
| `scripts/setup_demo_data.py` | 데모 데이터 | 인사관리 테이블 생성 + 50여 명 데이터 삽입 |

### server.py 상세

```python
class PostgresManager:
    """psycopg_pool.AsyncConnectionPool 을 래핑한 DB 접근 계층"""

    async def connect(self) -> None:
        """비동기 커넥션 풀 초기화"""

    async def close(self) -> None:
        """커넥션 풀 종료"""

    async def list_tables(self) -> List[Dict[str, Any]]:
        """information_schema에서 테이블 목록 조회"""

    async def describe_table(self, table_name, schema='public') -> List[Dict]:
        """테이블 컬럼 구조 조회"""

    async def run_query(self, sql: str) -> List[Dict[str, Any]]:
        """임의 SQL 실행 후 결과 반환"""


class MCPServer:
    """MCP Server 인스턴스. stdio 또는 SSE 모드로 실행"""

    def setup_handlers(self):
        """MCP 도구 등록 (query, list_tables, describe_table)"""

    async def run_stdio(self):
        """stdio 모드로 서버 실행 (Claude Desktop용)"""

    async def run_sse(self):
        """SSE 모드로 서버 실행 (Streamlit 웹 UI용)"""
```

### streamlit_app.py 상세

```python
# 데이터 표시
def display_data(df, show_chart=False, chart_type="bar", query=""):
    """DataFrame을 테이블 또는 차트로 표시"""

def display_chart(df, chart_type="bar", query=""):
    """Plotly 차트 생성 (pie, bar, line, area)"""

# SQL 생성
def generate_sql_from_nl(nl_query, table_list, model_name, chat_history=None):
    """자연어 → SQL 변환 (Claude/GPT)"""

def clean_sql_response(content: str) -> str:
    """AI 응답에서 SQL 추출 및 정제"""

# 차트 감지
def detect_chart_request(query: str) -> tuple[bool, str]:
    """쿼리에서 차트 유형 감지"""
```

---

## ⚙️ 설정 상세

### 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DB_CONNECTION` | (자동 조합) | 전체 연결 문자열 (우선 적용) |
| `DB_HOST` | `localhost` | PostgreSQL 호스트 |
| `DB_PORT` | `5432` | PostgreSQL 포트 |
| `DB_NAME` | `postgres` | 데이터베이스명 |
| `DB_USER` | `postgres` | 사용자명 |
| `DB_PASSWORD` | _(빈 값)_ | 비밀번호 |
| `MCP_SERVER_HOST` | `127.0.0.1` | SSE 서버 호스트 |
| `MCP_SERVER_PORT` | `8765` | SSE 서버 포트 |
| `ANTHROPIC_API_KEY` | - | Claude API 키 (Streamlit용) |
| `OPENAI_API_KEY` | - | OpenAI API 키 (Streamlit용) |
| `LOG_LEVEL` | `INFO` | 로깅 수준 |

---

## 🗃️ 데이터베이스 스키마

### 인사관리 테이블

```
┌────────────────────────────────────────────────────────────┐
│                         인사관리                              │
├────────────────────────────────────────────────────────────┤
│  사번         VARCHAR(20)   PK   직원 고유 코드 (YYYY+NNN)    │
│  이름         VARCHAR(50)        이름                        │
│  부서         VARCHAR(50)        소속 부서 (팀 단위)          │
│  직급         VARCHAR(50)        직급 (사원~팀장)             │
│  입사일       DATE               입사 날짜                    │
│  급여         INTEGER            월급여 (원)                  │
│  전화번호     VARCHAR(20)        연락처 (010-XXXX-XXXX)       │
│  이메일       VARCHAR(100)       회사 이메일                   │
│  성별         VARCHAR(10)        성별 (남/여)                 │
│  생년월일     DATE               생년월일                     │
└────────────────────────────────────────────────────────────┘
```

### 컬럼별 계산식

```sql
-- 근속연수 (컬럼 아님, 계산)
EXTRACT(YEAR FROM AGE(CURRENT_DATE, 입사일))

-- 연령 / 나이 (컬럼 아님, 계산)
EXTRACT(YEAR FROM AGE(CURRENT_DATE, 생년월일))
```

---

## 🔧 SQL 생성 규칙

### PostgreSQL 쿼리 패턴

```sql
-- CTE를 활용한 그룹 & 정렬 (ORDER BY 별칭 이슈 회피)
WITH cte AS (
    SELECT
        CASE WHEN ... THEN 'A' ELSE 'B' END AS label,
        CASE WHEN ... THEN 1 ELSE 2 END AS sort_key
    FROM 인사관리
)
SELECT label, COUNT(*) FROM cte
GROUP BY label, sort_key
ORDER BY sort_key;

-- 부서별 평균 근속연수
SELECT 부서,
       COUNT(*) AS 인원수,
       ROUND(AVG(EXTRACT(YEAR FROM AGE(CURRENT_DATE, 입사일))), 1) AS 평균근속연수
FROM 인사관리
GROUP BY 부서
ORDER BY 인원수 DESC
LIMIT 5;
```

---

## 📊 차트 키워드 매핑

| 키워드 | 차트 타입 | Plotly 함수 |
|--------|----------|-------------|
| 파이, 원그래프, 비율, 구성비 | `pie` | `px.pie()` |
| 막대, 바, bar | `bar` | `px.bar()` |
| 라인, 선, 추이, 추세 | `line` | `px.line()` |
| 영역, area | `area` | `px.area()` |

---

## 🔌 MCP 프로토콜

### stdio 모드 (Claude Desktop — 기본)

```
Claude Desktop                  Server
  │                               │
  │──── stdin: JSON-RPC ─────────►│  요청
  │◄─── stdout: JSON-RPC ─────────│  응답
  │                               │
```

### SSE 모드 (Streamlit — `--sse` 옵션)

```
Client                          Server
  │                               │
  │──── GET /sse ────────────────►│  SSE 연결 시작
  │◄─── SSE: endpoint event ──────│  메시지 엔드포인트 URL 반환
  │                               │
  │──── POST /messages ──────────►│  도구 호출 요청
  │◄─── SSE: message event ───────│  결과 스트리밍
  │                               │
```

---

## 🛡️ 기술 스택

| 분류 | 기술 | 버전 | 용도 |
|------|------|------|------|
| **언어** | Python | 3.10+ | 메인 언어 |
| **MCP** | mcp (SDK) | 1.26+ | Model Context Protocol |
| **DB 드라이버** | psycopg 3 | 3.2+ | PostgreSQL 비동기 접속 |
| **DB 풀링** | psycopg_pool | 3.2+ | AsyncConnectionPool |
| **웹 서버** | Starlette | 0.52+ | SSE 서버 |
| **ASGI** | Uvicorn | 0.40+ | ASGI 서버 |
| **UI** | Streamlit | 1.54+ | 웹 UI |
| **차트** | Plotly | 6.5+ | 인터랙티브 차트 |
| **데이터** | Pandas | 2.0+ | 데이터프레임 |
| **AI** | Anthropic | 0.79+ | Claude API |
| **AI** | OpenAI | 2.17+ | GPT API |
| **HTTP** | httpx | 0.28+ | SSE 클라이언트 |
| **환경** | python-dotenv | 1.0+ | 환경변수 로드 |

---

## 📋 API 참조

### MCP 도구 응답 형식

```python
# query — SQL 실행 결과 (JSON)
[{"column1": "value1", "column2": 123}, ...]

# list_tables — 테이블 목록
[{"table_schema": "public", "table_name": "인사관리"}, ...]

# describe_table — 컬럼 구조
[{"column_name": "사번", "data_type": "character varying",
  "is_nullable": "NO", "column_default": null}, ...]
```

### Streamlit 세션 상태

```python
st.session_state = {
    "messages": [
        {"role": "user", "content": "질문"},
        {"role": "assistant", "sql": "SELECT...", "data": DataFrame, ...}
    ],
    "selected_model": "claude-haiku-4-5-20251001",
    "sql_input": "마지막 실행 SQL"
}
```

---

## 📄 라이선스

MIT License

---

*Generated: 2026-02-09*
