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
DB_CONNECTION = os.getenv("DB_CONNECTION", "")


# =============================================================================
# HTTP Server Configuration
# =============================================================================
SERVER_HOST = os.getenv("SQLCL_SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SQLCL_SERVER_PORT", "8765"))
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


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
DEFAULT_MODEL = os.getenv("DEFAULT_AI_MODEL", "claude-opus-4-5-20251101")

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
    "page_title": "SQLcl AI Explorer",
    "page_icon": "📊",
    "layout": "wide"
}

PAGE_CONFIG_HTTP = {
    "page_title": "SQLcl AI Explorer (HTTP)",
    "page_icon": "⚡",
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
        * IBSA_DATE (Join Date) - USE THIS FOR TENURE CALCULATION
        * TESA_DATE (Resignation Date) - NULL means currently employed
        * SEX_GUBN (Gender)
        * BRTH_DATE (Birth Date)
        * EMPL_DUTY (Position/Title Code, FK to INTONG 150xx)
        * EMPL_JKGB (Job Grade Code, FK to INTONG 151xx)
    
    - HRM_DEPT (Department Master):
        * DEPT_CD (Department Code, PK)
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
- ALWAYS use table aliases (e.g., INMAST M, HRM_DEPT D, INTONG T).
- ALWAYS prefix column names with table alias.
- For tenure/service years: ALWAYS use ROUND(MONTHS_BETWEEN(SYSDATE, M.IBSA_DATE) / 12, 1) to show 1 decimal place.
- For any decimal/float results: ALWAYS use ROUND(..., 1) to limit to 1 decimal place.
- Join INMAST and HRM_DEPT: M.DEPA_CODE = D.DEPT_CD
- Join INMAST and INTONG for position name: M.EMPL_DUTY = T.TONG_CODE
- Use M.EMPL_NUMB for counting employees.
- Use FETCH FIRST n ROWS ONLY for limits.
- For department name, use D.DEPT_NM from HRM_DEPT table.
- For position/title name, join INTONG: SELECT T.TONG_1NAM FROM INTONG T WHERE M.EMPL_DUTY = T.TONG_CODE
- For currently employed: WHERE M.TESA_DATE IS NULL
- For manager/leader positions: WHERE M.EMPL_DUTY IN ('15001','15002','15003','15004','15005','15006','15007','150A2','150A3')
"""


# =============================================================================
# Chart Configuration
# =============================================================================
CHART_KEYWORDS = ['차트', '그래프', 'chart', 'graph', '시각화', 'visualize', 'plot', '그려', '보여줘', '표시']
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
