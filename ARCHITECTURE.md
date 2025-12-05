# SQLcl MCP Server - 기술 구조 및 사양

> 버전: 2.1.0 | 최종 수정: 2025-12-03

---

## 📐 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client Layer                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Streamlit UI   │  │  Claude Desktop │  │  VS Code        │              │
│  │  (Port 8501)    │  │  (stdio)        │  │  MCP Client     │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
│           │                    │                    │                        │
│           │ SSE/HTTP           │ stdio              │ SSE/HTTP              │
│           ▼                    ▼                    ▼                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                              MCP Server Layer                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    MCP Server (server.py)                           │    │
│  │  ┌─────────────────┐              ┌─────────────────┐               │    │
│  │  │  SSE Transport  │              │ stdio Transport │               │    │
│  │  │  (기본 모드)     │              │ (--stdio 옵션)  │               │    │
│  │  │  Port: 8765     │              │                 │               │    │
│  │  └─────────────────┘              └─────────────────┘               │    │
│  │                                                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                    MCP Tools                                 │    │    │
│  │  │  • execute_sql      - SQL 쿼리 실행                          │    │    │
│  │  │  • get_tables       - 테이블 목록 조회                        │    │    │
│  │  │  • describe_table   - 테이블 구조 조회                        │    │    │
│  │  │  • get_status       - 연결 상태 확인                          │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│                              Session Layer                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              SQLcl Persistent Session (SQLclSession)                │    │
│  │  • subprocess.Popen으로 SQLcl 프로세스 유지                         │    │
│  │  • stdin/stdout 파이프로 명령 전송/결과 수신                         │    │
│  │  • CSV 형식 출력 (SET SQLFORMAT csv)                                │    │
│  │  • 타임아웃: 60초 (설정 가능)                                        │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   │ JDBC                                     │
│                                   ▼                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                              Database Layer                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         Oracle Database                              │    │
│  │  • Connection: user/password@host:port/service                       │    │
│  │  • 주요 테이블: INMAST, ZME, HRM_DEPT, INTONG                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 데이터 흐름

### 1. 자연어 → SQL 변환 흐름

```
┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ 사용자    │───►│ Streamlit UI │───►│ AI API      │───►│ SQL 생성     │
│ 자연어    │    │ (입력 처리)  │    │ Claude/GPT  │    │ (정제된 SQL) │
└──────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

### 2. SQL 실행 흐름

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Streamlit UI │───►│ MCP Client   │───►│ MCP Server   │───►│ SQLcl Session│
│ (SSE 요청)   │    │ (httpx)      │    │ (Starlette)  │    │ (subprocess) │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                                                                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 차트/테이블   │◄───│ DataFrame    │◄───│ CSV 파싱     │◄───│ Oracle DB    │
│ (Plotly)     │    │ (Pandas)     │    │              │    │ (결과 반환)  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

---

## 📦 모듈 구조

### 핵심 파일

| 파일 | 역할 | 주요 클래스/함수 |
|------|------|------------------|
| `server.py` | MCP 서버 | `SQLclSession`, `execute_sql()`, `run_sse_server()` |
| `streamlit_app.py` | 웹 UI | `display_data()`, `display_chart()`, `generate_sql_from_nl()` |
| `config.py` | 설정 관리 | 환경변수, DB 스키마, SQL 생성 규칙 |

### server.py 상세

```python
class SQLclSession:
    """SQLcl 영속 세션 관리"""
    
    def __init__(self):
        self.process = None      # subprocess.Popen 객체
        self.connected = False   # 연결 상태
        
    def connect(self) -> bool:
        """SQLcl 프로세스 시작 및 DB 연결"""
        
    def execute(self, sql: str, timeout: int = 60) -> str:
        """SQL 실행 및 결과 반환 (CSV 형식)"""
        
    def disconnect(self):
        """세션 종료"""

# MCP 도구
@mcp.tool()
def execute_sql(sql: str) -> str: ...

@mcp.tool()
def get_tables() -> str: ...

@mcp.tool()
def describe_table(table_name: str) -> str: ...

@mcp.tool()
def get_status() -> str: ...
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

def extract_chart_title(query: str) -> str:
    """쿼리에서 차트 제목 추출"""
```

---

## ⚙️ 설정 상세

### 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DB_CONNECTION` | - | 전체 연결 문자열 (우선) |
| `DB_HOST` | `localhost` | 호스트 |
| `DB_PORT` | `11521` | 포트 |
| `DB_SERVICE` | `ORCL` | 서비스명 |
| `DB_USER` | - | 사용자 |
| `DB_PASSWORD` | - | 비밀번호 |
| `SQLCL_PATH` | `C:\...\sql.exe` | SQLcl 경로 |
| `MCP_SERVER_HOST` | `127.0.0.1` | 서버 호스트 |
| `MCP_SERVER_PORT` | `8765` | 서버 포트 |
| `ANTHROPIC_API_KEY` | - | Claude API 키 |
| `OPENAI_API_KEY` | - | OpenAI API 키 |
| `DEFAULT_AI_MODEL` | `claude-haiku-4-5-20251001` | 기본 AI 모델 |
| `SQLCL_TIMEOUT` | `60` | SQL 실행 타임아웃(초) |

### SQLcl 초기화 설정

```sql
SET PAGESIZE 50000
SET LINESIZE 32767
SET LONG 50000
SET LONGCHUNKSIZE 50000
SET TRIMSPOOL ON
SET TRIMOUT ON
SET FEEDBACK OFF
SET HEADING ON
SET SQLFORMAT csv
```

---

## 🗃️ 데이터베이스 스키마

### 주요 테이블

```
┌─────────────────────────────────────────────────────────────────────┐
│                           INMAST (직원 마스터)                        │
├─────────────────────────────────────────────────────────────────────┤
│  EMPL_NUMB    VARCHAR2   PK   사번                                   │
│  EMPL_NAME    VARCHAR2        이름                                   │
│  DEPA_CODE    VARCHAR2   FK   부서코드 → ZME.DEPA_CODE               │
│  IBSA_DATE    VARCHAR2        입사일 (YYYYMMDD)                      │
│  TESA_DATE    VARCHAR2        퇴사일 (YYYYMMDD, NULL=재직)           │
│  SEX_GUBN     VARCHAR2        성별 ('1'=남, '2'=여)                  │
│  BRTH_DATE    VARCHAR2        생년월일 (YYYYMMDD)                    │
│  EMPL_DUTY    VARCHAR2   FK   직위코드 → INTONG (150xx)              │
│  EMPL_JKGB    VARCHAR2   FK   직급코드 → INTONG (151xx)              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                           ZME (부서 마스터)                           │
├─────────────────────────────────────────────────────────────────────┤
│  DEPA_CODE    VARCHAR2   PK   부서코드                               │
│  DEPA_NAME    VARCHAR2        부서명                                 │
│  PRNT_NAME    VARCHAR2        상위부서명                             │
│  ORGA_SYST    VARCHAR2        조직체계                               │
│  APPL_DATE    VARCHAR2        적용일자                               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         INTONG (코드 마스터)                          │
├─────────────────────────────────────────────────────────────────────┤
│  TONG_CODE    VARCHAR2   PK   코드                                   │
│  TONG_SECT    VARCHAR2        카테고리 ('150'=직위, '151'=직급)      │
│  TONG_DETA    VARCHAR2        상세코드                               │
│  TONG_1NAM    VARCHAR2        코드명                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 테이블 관계

```
INMAST ─────┬───── ZME (부서)
  │         │      M.DEPA_CODE = Z.DEPA_CODE
  │         │
  │         └───── INTONG (직위)
  │                M.EMPL_DUTY = T.TONG_CODE
  │
  └─────────────── INTONG (직급)
                   M.EMPL_JKGB = T.TONG_CODE
```

---

## 🔧 SQL 생성 규칙

### 날짜 처리

```sql
-- 안전한 날짜 변환 (DEFAULT NULL ON CONVERSION ERROR)
TO_DATE(M.IBSA_DATE DEFAULT NULL ON CONVERSION ERROR, 'YYYYMMDD')

-- 근속연수 계산
ROUND(MONTHS_BETWEEN(SYSDATE, 
  TO_DATE(M.IBSA_DATE DEFAULT NULL ON CONVERSION ERROR, 'YYYYMMDD')
) / 12, 1) AS AVG_TENURE

-- 나이 계산
TRUNC(MONTHS_BETWEEN(SYSDATE, 
  TO_DATE(M.BRTH_DATE DEFAULT NULL ON CONVERSION ERROR, 'YYYYMMDD')
) / 12) AS AGE
```

### 그룹 쿼리 패턴

```sql
-- 연령대별 (서브쿼리 패턴)
SELECT AGE_GROUP, COUNT(*) AS EMP_COUNT 
FROM (
  SELECT CASE 
    WHEN AGE < 30 THEN '20대' 
    WHEN AGE < 40 THEN '30대' 
    WHEN AGE < 50 THEN '40대' 
    WHEN AGE < 60 THEN '50대' 
    ELSE '60대 이상' 
  END AS AGE_GROUP
  FROM (
    SELECT TRUNC(MONTHS_BETWEEN(SYSDATE, 
      TO_DATE(M.BRTH_DATE DEFAULT NULL ON CONVERSION ERROR, 'YYYYMMDD')
    ) / 12) AS AGE 
    FROM INMAST M 
    WHERE M.BRTH_DATE IS NOT NULL
  )
) 
GROUP BY AGE_GROUP 
ORDER BY DECODE(AGE_GROUP, '20대', 1, '30대', 2, '40대', 3, '50대', 4, 5)

-- 근속연수 구간별 (서브쿼리 패턴)
SELECT TENURE_GROUP, COUNT(*) AS EMP_COUNT 
FROM (
  SELECT 
    CASE 
      WHEN TENURE < 1 THEN '1년 미만' 
      WHEN TENURE < 3 THEN '1~3년' 
      WHEN TENURE < 5 THEN '3~5년' 
      WHEN TENURE < 10 THEN '5~10년' 
      WHEN TENURE < 15 THEN '10~15년' 
      WHEN TENURE < 20 THEN '15~20년' 
      ELSE '20년 이상' 
    END AS TENURE_GROUP,
    CASE 
      WHEN TENURE < 1 THEN 1 
      WHEN TENURE < 3 THEN 2 
      WHEN TENURE < 5 THEN 3 
      WHEN TENURE < 10 THEN 4 
      WHEN TENURE < 15 THEN 5 
      WHEN TENURE < 20 THEN 6 
      ELSE 7 
    END AS SORT_ORDER
  FROM (
    SELECT ROUND(MONTHS_BETWEEN(SYSDATE, 
      TO_DATE(M.IBSA_DATE DEFAULT NULL ON CONVERSION ERROR, 'YYYYMMDD')
    ) / 12, 1) AS TENURE 
    FROM INMAST M 
    WHERE M.IBSA_DATE IS NOT NULL
  )
) 
GROUP BY TENURE_GROUP, SORT_ORDER 
ORDER BY SORT_ORDER
```

---

## 📊 차트 설정

### 차트 키워드 매핑

| 키워드 | 차트 타입 | Plotly 함수 |
|--------|----------|-------------|
| 파이, 원그래프, 비율, 구성비 | `pie` | `px.pie()` |
| 막대, 바, bar | `bar` | `px.bar()` |
| 라인, 선, 추이, 추세 | `line` | `px.line()` |
| 영역, area | `area` | `px.area()` |

### Plotly 설정

```python
# 데이터 순서 유지
fig.update_xaxes(
    categoryorder='array', 
    categoryarray=df[label_col].tolist()
)

# 고유 key로 차트 ID 충돌 방지
st.plotly_chart(fig, use_container_width=True, key=f"chart_{uuid.uuid4().hex[:8]}")
```

---

## 🔌 MCP 프로토콜

### SSE 모드 (기본)

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

### stdio 모드

```
Claude Desktop                  Server
  │                               │
  │──── stdin: JSON-RPC ─────────►│  요청
  │◄─── stdout: JSON-RPC ─────────│  응답
  │                               │
```

---

## 🛡️ 기술 스택 상세

| 분류 | 기술 | 버전 | 용도 |
|------|------|------|------|
| **언어** | Python | 3.10+ | 메인 언어 |
| **MCP** | mcp | 1.0.0+ | Model Context Protocol |
| **웹 서버** | Starlette | 0.27+ | SSE 서버 |
| **ASGI** | Uvicorn | 0.23+ | ASGI 서버 |
| **HTTP** | httpx | 0.24+ | SSE 클라이언트 |
| **UI** | Streamlit | 1.28+ | 웹 UI |
| **차트** | Plotly | 5.18+ | 인터랙티브 차트 |
| **데이터** | Pandas | 2.0+ | 데이터프레임 |
| **AI** | Anthropic | 0.18+ | Claude API |
| **AI** | OpenAI | 1.0+ | GPT API |
| **DB** | SQLcl | 24.1+ | Oracle CLI |
| **환경** | python-dotenv | 1.0+ | 환경변수 로드 |

---

## 📋 API 참조

### MCP 도구 응답 형식

```python
# execute_sql
{
    "success": True/False,
    "data": "CSV 형식 결과" | "ERROR: 메시지"
}

# get_tables
"TABLE_NAME\ntable1\ntable2\n..."

# describe_table
"COLUMN_NAME,DATA_TYPE,NULLABLE\ncol1,VARCHAR2,Y\n..."

# get_status
"Connected to: user@host:port/service" | "Not connected"
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

*Generated: 2025-12-03*
