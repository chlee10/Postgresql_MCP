"""
PostgreSQL AI Explorer - Configuration

모든 설정값을 한 곳에서 관리합니다.
환경 변수로 오버라이드 가능합니다.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)


# =============================================================================
# Database Configuration
# =============================================================================
# PostgreSQL Connection Settings
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# Connection String (optional override)
# Format: postgresql://user:password@host:port/dbname
DB_CONNECTION = os.getenv("DB_CONNECTION")

if not DB_CONNECTION:
    if DB_PASSWORD:
        DB_CONNECTION = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    else:
        DB_CONNECTION = f"postgresql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# =============================================================================
# MCP Server Configuration
# =============================================================================
SERVER_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8765"))
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# =============================================================================
# Streamlit Configuration
# =============================================================================
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# =============================================================================
# AI Configuration
# =============================================================================
# Models
AI_MODELS = [
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5-20251101",
    "gpt-5",
    "gpt-5-1",
    "gpt-5-2"
]

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

MODEL_DISPLAY = {
    "claude-haiku-4-5-20251001": "Claude 4.5 Haiku ",
    "claude-sonnet-4-5-20250929": "Claude 4.5 Sonnet ",
    "claude-opus-4-5-20251101": "Claude 4.5 Opus ",
    "gpt-5": "GPT-5",
    "gpt-5-1": "GPT-5-1",
    "gpt-5-2": "GPT-5-2"
}

# Streamlit Page Config
PAGE_CONFIG = {
    "page_title": "PostgreSQL AI Explorer",
    "page_icon": "🐘",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# App Version
APP_VERSION = "1.0.0"

# Chart Keywords (for auto-viz)
CHART_KEYWORDS = ["차트", "그래프", "시각화", "보여줘", "그려줘", "chart", "plot", "graph"]
LINE_CHART_KEYWORDS = ["추세", "변화", "흐름", "라인", "선", "trend", "line", "time"]
PIE_CHART_KEYWORDS = ["비율", "비중", "점유율", "파이", "원형", "ratio", "share", "pie"]
AREA_CHART_KEYWORDS = ["누적", "영역", "면적", "area", "stack"]

# =============================================================================
# Prompts
# =============================================================================

DB_SCHEMA_INFO = """
PostgreSQL database.
Use standardized SQL.
"""

SQL_GENERATION_RULES = """
1. Return ONLY the SQL query.
2. Use valid PostgreSQL syntax.
3. Limit results to 100 rows unless specified otherwise.
4. IMPORTANT: Only use table names provided in the "Available Tables" list. Do NOT hallucinate table names like 'employees' or 'departments' if they are not listed.
5. Schema/Calculated Columns Hints:
   - Table '인사관리' has columns: id, 이름, 부서, 직급, 입사일, 급여, 전화번호, 이메일, 성별, 생년월일.
   - '근속연수' (Years of Service) is NOT a column. Calculate it as: EXTRACT(YEAR FROM AGE(CURRENT_DATE, 입사일))
   - '연령'/'나이' (Age) is NOT a column. Calculate it as: EXTRACT(YEAR FROM AGE(CURRENT_DATE, 생년월일))
6. PREFER Common Table Expressions (WITH clauses) for complex groupings/orderings.
   - CRITICAL: Do not use column aliases in ORDER BY CASE statements. It fails in PostgreSQL.
   - Solution: Calculate both the label AND a numeric sort_key in a CTE, then group by both.
   - Example: 
     WITH cte AS (
         SELECT CASE WHEN ... THEN 'A' ELSE 'B' END as label,
                CASE WHEN ... THEN 1 ELSE 2 END as sort_key
         FROM table
     )
     SELECT label, count(*) FROM cte GROUP BY label, sort_key ORDER BY sort_key
"""
