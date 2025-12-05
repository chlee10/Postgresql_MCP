"""
SQLcl AI Explorer - Configuration

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
SQLCL_PATH = os.getenv("SQLCL_PATH", r"C:\Users\chiho\sqlcl\bin\sql.exe")

# Oracle Connection - 개별 설정 또는 전체 연결 문자열 사용
# 방법 1: 개별 설정 (DB_CONNECTION이 비어있을 때 사용)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "11521")
DB_SERVICE = os.getenv("DB_SERVICE", "ORCL")  # 또는 SID
DB_USER = os.getenv("DB_USER", "")  # .env에서 설정
DB_PASSWORD = os.getenv("DB_PASSWORD", "")  # .env에서 설정

# 방법 2: 전체 연결 문자열 (우선 사용)
# 형식: user/password@host:port/service 또는 user/password@tnsname
DB_CONNECTION = os.getenv("DB_CONNECTION", "")

# 연결 문자열 생성 (DB_CONNECTION이 비어있으면 개별 설정으로 조합)
if not DB_CONNECTION and DB_USER and DB_PASSWORD:
    DB_CONNECTION = f"{DB_USER}/{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_SERVICE}"


# =============================================================================
# MCP Server Configuration
# =============================================================================
SERVER_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8765"))
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


# =============================================================================
# Streamlit Configuration
# =============================================================================
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8503"))


# =============================================================================
# SQLcl Settings
# =============================================================================
SQLCL_TIMEOUT = int(os.getenv("SQLCL_TIMEOUT", "60"))
SQLCL_INIT_COMMANDS = """SET PAGESIZE 50000
SET LINESIZE 32767
SET LONG 50000
SET LONGCHUNKSIZE 50000
SET TRIMSPOOL ON
SET TRIMOUT ON
SET FEEDBACK OFF
SET HEADING ON
SET SQLFORMAT csv
"""


# =============================================================================
# AI Model Configuration
# =============================================================================
DEFAULT_MODEL = os.getenv("DEFAULT_AI_MODEL", "claude-haiku-4-5-20251001")

# API Keys (민감 정보 - .env 파일에서 설정)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

AI_MODELS = [
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5-20251101",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]

MODEL_DISPLAY = {
    "claude-sonnet-4-5-20250929": ("🟣", "Claude Sonnet 4.5", "균형잡힌 성능"),
    "claude-haiku-4-5-20251001": ("🟢", "Claude Haiku 4.5", "빠른 응답"),
    "claude-opus-4-5-20251101": ("🔵", "Claude Opus 4.5", "최고 성능"),
    "gpt-4o": ("🟡", "GPT-4o", "고성능"),
    "gpt-4-turbo": ("🟠", "GPT-4 Turbo", "안정적"),
    "gpt-3.5-turbo": ("⚪", "GPT-3.5 Turbo", "경제적"),
}


# =============================================================================
# Streamlit Page Config
# =============================================================================
PAGE_CONFIG = {
    "page_title": "Oracle MCP Server",
    "page_icon": "🔶",
    "layout": "wide"
}


# =============================================================================
# Database Schema Info (for AI prompts)
# =============================================================================
DB_SCHEMA_INFO = """
Important Tables and Columns:
    - INMAST (Employee Master): 
        * EMPL_NUMB (Employee ID, PK)
        * EMPL_NAME (Name)
        * DEPA_CODE (Dept Code, FK to HRM_DEPT)
        * IBSA_DATE (Join Date, VARCHAR 'YYYYMMDD') - USE THIS FOR TENURE CALCULATION
        * TESA_DATE (Resignation Date, VARCHAR 'YYYYMMDD') - NULL means currently employed
        * SEX_GUBN (Gender: '1'=Male, '2'=Female)
        * BRTH_DATE (Birth Date, VARCHAR 'YYYYMMDD' format, e.g., '19850315')
        * EMPL_DUTY (Position/Title Code, FK to INTONG 150xx)
        * EMPL_JKGB (Job Grade Code, FK to INTONG 151xx)
    
    - ZME (Department Master - Main):
        * DEPA_CODE (Department Code, PK) - INMAST.DEPA_CODE 와 직접 조인 가능!
        * DEPA_NAME (Department Name)
        * PRNT_NAME (Parent Department Name)
        * ORGA_SYST (Organization System)
        * APPL_DATE (Apply Date)
        
    - HRM_DEPT (Department Master - HR System):
        * DEPT_CD (Department Code, PK) - 주의: INMAST.DEPA_CODE와 형식이 다름!
        * DEPT_NM (Department Name)
        * UPP_DEPT_CD (Parent Department Code)
    
    - INTONG (Code Master):
        * TONG_CODE (Code, PK)
        * TONG_SECT (Category)
        * TONG_DETA (Detail Code)
        * TONG_1NAM (Code Name)
        
    - Position Codes (EMPL_DUTY - TONG_SECT='150'):
        * 15001=본부장, 15002=센터장, 15003=처장, 15004=부장
        * 15005=차장, 15006=팀장, 15007=단장, 15008=담당
        * 150A2=국장, 150A3=실장, 150A4=팀원
        
    - Job Grade Codes (EMPL_JKGB - TONG_SECT='151'):
        * 15111=수석연구위원, 15112=선임연구위원, 15113=연구위원
        * 15114=부연구위원, 15136=책임행정원, 15137=선임행정원
        * 15138=행정원, 151AA=책임연구원, 151AB=연구원
"""

SQL_GENERATION_RULES = """
Rules:
- Return ONLY the SQL query without any explanation or description.
- NEVER include any text before or after the SQL query.
- Use standard Oracle syntax.
- ALWAYS use ENGLISH column aliases (e.g., DEPT_NAME, EMP_COUNT, AVG_TENURE). NEVER use Korean aliases.
- ALWAYS use table aliases (e.g., INMAST M, ZME Z, INTONG T).
- ALWAYS prefix column names with table alias.
- **DATE FIELDS ARE VARCHAR 'YYYYMMDD'**: IBSA_DATE, TESA_DATE, BRTH_DATE are stored as VARCHAR in 'YYYYMMDD' format.
- **SAFE DATE CONVERSION**: Always use TO_DATE with DEFAULT NULL ON CONVERSION ERROR: TO_DATE(M.BRTH_DATE DEFAULT NULL ON CONVERSION ERROR, 'YYYYMMDD')
- For tenure/service years: ROUND(MONTHS_BETWEEN(SYSDATE, TO_DATE(M.IBSA_DATE DEFAULT NULL ON CONVERSION ERROR, 'YYYYMMDD')) / 12, 1) AS AVG_TENURE (returns NUMBER, not VARCHAR)
- For age calculation: TRUNC(MONTHS_BETWEEN(SYSDATE, TO_DATE(M.BRTH_DATE DEFAULT NULL ON CONVERSION ERROR, 'YYYYMMDD')) / 12)
- For age group filtering (40대 이상): TO_DATE(M.BRTH_DATE DEFAULT NULL ON CONVERSION ERROR, 'YYYYMMDD') IS NOT NULL AND TRUNC(MONTHS_BETWEEN(SYSDATE, TO_DATE(M.BRTH_DATE DEFAULT NULL ON CONVERSION ERROR, 'YYYYMMDD')) / 12) >= 40
- **AGE GROUP QUERIES**: When grouping by age (연령대별), use subquery pattern to avoid GROUP BY issues:
  SELECT AGE_GROUP, COUNT(*) AS EMP_COUNT FROM (
    SELECT CASE WHEN AGE < 30 THEN '20대' WHEN AGE < 40 THEN '30대' WHEN AGE < 50 THEN '40대' WHEN AGE < 60 THEN '50대' ELSE '60대 이상' END AS AGE_GROUP
    FROM (SELECT TRUNC(MONTHS_BETWEEN(SYSDATE, TO_DATE(M.BRTH_DATE DEFAULT NULL ON CONVERSION ERROR, 'YYYYMMDD')) / 12) AS AGE FROM INMAST M WHERE M.BRTH_DATE IS NOT NULL)
  ) GROUP BY AGE_GROUP ORDER BY DECODE(AGE_GROUP, '20대', 1, '30대', 2, '40대', 3, '50대', 4, 5)
- **TENURE GROUP QUERIES**: When grouping by tenure/service years (근속연수 구간별), use subquery pattern:
  SELECT TENURE_GROUP, COUNT(*) AS EMP_COUNT FROM (
    SELECT CASE WHEN TENURE < 1 THEN '1년 미만' WHEN TENURE < 3 THEN '1~3년' WHEN TENURE < 5 THEN '3~5년' WHEN TENURE < 10 THEN '5~10년' WHEN TENURE < 15 THEN '10~15년' WHEN TENURE < 20 THEN '15~20년' ELSE '20년 이상' END AS TENURE_GROUP,
           CASE WHEN TENURE < 1 THEN 1 WHEN TENURE < 3 THEN 2 WHEN TENURE < 5 THEN 3 WHEN TENURE < 10 THEN 4 WHEN TENURE < 15 THEN 5 WHEN TENURE < 20 THEN 6 ELSE 7 END AS SORT_ORDER
    FROM (SELECT ROUND(MONTHS_BETWEEN(SYSDATE, TO_DATE(M.IBSA_DATE DEFAULT NULL ON CONVERSION ERROR, 'YYYYMMDD')) / 12, 1) AS TENURE FROM INMAST M WHERE M.IBSA_DATE IS NOT NULL)
  ) GROUP BY TENURE_GROUP, SORT_ORDER ORDER BY SORT_ORDER
- **NUMERIC ORDER BY**: When ordering by numeric columns (counts, averages, years), use the numeric column directly in ORDER BY, NOT a string-formatted alias. Example: ORDER BY ROUND(...) DESC, not ORDER BY formatted_string DESC
- **IMPORTANT**: Join INMAST and ZME for department: M.DEPA_CODE = Z.DEPA_CODE (NOT HRM_DEPT!)
- Join INMAST and INTONG for position name: M.EMPL_DUTY = T.TONG_CODE
- Use M.EMPL_NUMB for counting employees.
- Use FETCH FIRST n ROWS ONLY for limits.
- For department name, use Z.DEPA_NAME from ZME table.
- For position/title name, join INTONG: SELECT T.TONG_1NAM FROM INTONG T WHERE M.EMPL_DUTY = T.TONG_CODE
- For manager/leader positions: WHERE M.EMPL_DUTY IN ('15001','15002','15003','15004','15005','15006','15007','150A2','150A3')
- **CRITICAL**: Do NOT filter by TESA_DATE unless user explicitly asks for "재직자" or "현재 직원". Include ALL employees by default.
- **UNION with ORDER BY**: When using UNION/UNION ALL, do NOT add ORDER BY after UNION. Instead, include a RANK_TYPE or SORT_ORDER column in each SELECT and let the natural order work. Example for Top/Bottom N:
  SELECT DEPT_NAME, EMP_COUNT, 1 AS SORT_ORDER FROM (SELECT ... ORDER BY EMP_COUNT DESC FETCH FIRST 5 ROWS ONLY)
  UNION ALL
  SELECT DEPT_NAME, EMP_COUNT, 2 AS SORT_ORDER FROM (SELECT ... ORDER BY EMP_COUNT ASC FETCH FIRST 5 ROWS ONLY)
  -- No ORDER BY after UNION, use SORT_ORDER column for display grouping
- For gender counts by group: Use SUM(CASE WHEN M.SEX_GUBN = '1' THEN 1 ELSE 0 END) AS MALE_COUNT, SUM(CASE WHEN M.SEX_GUBN = '2' THEN 1 ELSE 0 END) AS FEMALE_COUNT
"""


# =============================================================================
# Chart Configuration
# =============================================================================
CHART_KEYWORDS = ['차트', '그래프', 'chart', 'graph', '시각화', 'visualize', 'plot', '그려줘', '그려', '막대', '라인', '파이', '원형', 'bar', 'line', 'pie']
BAR_CHART_KEYWORDS = ['막대', 'bar', '바']
LINE_CHART_KEYWORDS = ['라인', 'line', '선', '추이', '추세', 'trend']
PIE_CHART_KEYWORDS = ['파이', 'pie', '원형', '원그래프', '비율', '구성비', '도넛', 'donut']
AREA_CHART_KEYWORDS = ['area', '영역', '면적']


# =============================================================================
# Logging Configuration
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# =============================================================================
# App Version
# =============================================================================
APP_VERSION = "2.1.0"
