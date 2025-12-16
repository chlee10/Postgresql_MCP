import streamlit as st
import subprocess
import os
import sys
import asyncio
import pandas as pd
import re
import tempfile
import time
import logging
from io import StringIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Optional imports for MCP
try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
except ImportError:
    ClientSession = None
    stdio_client = None
    StdioServerParameters = None

# Optional imports for OpenAI
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Optional imports for Anthropic (Claude)
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# --- Configuration ---
st.set_page_config(
    page_title="SQLcl AI Explorer",
    page_icon="📊",
    layout="wide"
)

# Load config from environment variables
SQLCL_PATH = os.getenv("SQLCL_PATH", r"C:\Users\chiho\sqlcl\bin\sql.exe")
DB_CONNECTION = os.getenv("DB_CONNECTION", "")

# --- Session State Initialization ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'sql_input' not in st.session_state:
    st.session_state.sql_input = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "execution_mode" not in st.session_state:
    st.session_state.execution_mode = "Direct (Fast)"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "claude-sonnet-4-5-20250929"

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sqlcl-client")

# --- File-based SQL Client Class (Stable) ---
class SQLClient:
    """
    파일 기반 SQLcl 클라이언트.
    매 쿼리마다 임시 SQL 파일을 생성하고 subprocess.run()으로 실행합니다.
    버퍼링 문제를 완전히 회피하여 안정적인 실행을 보장합니다.
    """
    
    def __init__(self, sqlcl_path, db_connection):
        self.sqlcl_path = sqlcl_path
        self.db_connection = db_connection
        self._connection_tested = False
        self._test_connection()
    
    def _get_env(self):
        """SQLcl 실행에 필요한 환경 변수를 반환합니다."""
        env = os.environ.copy()
        env["NLS_LANG"] = "KOREAN_KOREA.AL32UTF8"
        env["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8"
        return env
    
    def _test_connection(self):
        """연결 테스트를 수행합니다."""
        try:
            success, result = self.run_query("SELECT 1 FROM DUAL")
            self._connection_tested = success
            if success:
                logger.info("SQLcl connection test successful")
            else:
                logger.warning(f"SQLcl connection test failed: {result}")
        except Exception as e:
            logger.error(f"SQLcl connection test error: {e}")
            self._connection_tested = False
    
    def run_query(self, query, timeout=60):
        """
        SQL 쿼리를 실행하고 결과를 반환합니다.
        
        Args:
            query: 실행할 SQL 쿼리
            timeout: 타임아웃 (초), 기본값 60초
            
        Returns:
            (success: bool, result: str) 튜플
        """
        query = query.strip()
        if not query.endswith(";"):
            query += ";"
        
        # 임시 SQL 파일 생성
        sql_content = f"""SET PAGESIZE 50000
SET LINESIZE 32767
SET LONG 50000
SET LONGCHUNKSIZE 50000
SET TRIMSPOOL ON
SET TRIMOUT ON
SET FEEDBACK OFF
SET HEADING ON
SET SQLFORMAT csv

{query}

EXIT;
"""
        
        sql_file = None
        try:
            # 임시 파일 생성 (삭제하지 않음 - finally에서 처리)
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.sql', 
                delete=False, 
                encoding='utf-8'
            ) as f:
                f.write(sql_content)
                sql_file = f.name
            
            logger.info(f"Executing query via file: {sql_file}")
            logger.debug(f"Query: {query[:100]}...")
            
            start_time = time.time()
            
            # SQLcl 실행
            result = subprocess.run(
                [self.sqlcl_path, "-S", self.db_connection, f"@{sql_file}"],
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace',
                env=self._get_env()
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Query completed in {elapsed:.2f}s, returncode={result.returncode}")
            
            # 결과 처리
            stdout = result.stdout.strip() if result.stdout else ""
            stderr = result.stderr.strip() if result.stderr else ""
            
            # JAVA_TOOL_OPTIONS 메시지 필터링
            if stderr:
                stderr_lines = [
                    line for line in stderr.split('\n') 
                    if not line.startswith('Picked up JAVA_TOOL_OPTIONS')
                ]
                stderr = '\n'.join(stderr_lines).strip()
            
            # stdout에서 에러 메시지 분리 (ORA-, SP2- 등)
            # CSV 데이터만 추출하고 에러 메시지는 분리
            output_lines = stdout.split('\n')
            csv_lines = []
            error_lines = []
            in_error_block = False  # 에러 블록 내부인지 추적
            
            for line in output_lines:
                line_stripped = line.strip()
                
                # 에러 블록 시작 감지
                if any(err in line_stripped for err in ['ORA-', 'SP2-', 'Error at']):
                    in_error_block = True
                    error_lines.append(line)
                    continue
                
                # 에러 블록 내부 라인 (Oracle 에러 설명)
                if in_error_block:
                    # 에러 설명 관련 키워드
                    if any(kw in line_stripped for kw in ['*Cause:', '*Action:', '*Params:', 'More Details', 
                                                           'https://docs.oracle', '1)', '2)', '3)', '4)']):
                        error_lines.append(line)
                        continue
                    # 들여쓰기된 설명 라인
                    if line.startswith('       ') or line.startswith('\t'):
                        error_lines.append(line)
                        continue
                    # 빈 줄이면 에러 블록 종료 가능
                    if not line_stripped:
                        continue
                    # 새로운 데이터 시작으로 간주
                    in_error_block = False
                
                # 에러 관련 라인 필터링
                if any(err in line_stripped for err in ['ORA-', 'SP2-', 'Error', '오류', 'https://docs.oracle']):
                    error_lines.append(line)
                # 빈 줄이나 메타 정보 스킵
                elif line_stripped in ['', 'Execution successful.', 'Commit complete.']:
                    continue
                # 파일 경로 정보 스킵
                elif '파일 @' in line_stripped or '명령 -' in line_stripped or '명령행 오류' in line_stripped:
                    error_lines.append(line)
                else:
                    csv_lines.append(line)
            
            # 에러가 있으면 실패로 처리
            if error_lines and not csv_lines:
                return False, '\n'.join(error_lines)
            
            # CSV 데이터 반환 (에러가 있어도 데이터가 있으면 성공으로 처리)
            stdout = '\n'.join(csv_lines)
            
            # 에러 체크
            if result.returncode != 0 and not stdout:
                error_msg = stderr or stdout or f"SQLcl exited with code {result.returncode}"
                return False, error_msg
            
            # Oracle 에러 체크
            if stdout.startswith("ORA-") or "SP2-" in stdout or "Error" in stdout[:50]:
                return False, stdout
            
            # 빈 결과 체크
            if not stdout:
                return True, ""
            
            return True, stdout
            
        except subprocess.TimeoutExpired:
            logger.error(f"Query timed out after {timeout}s")
            return False, f"Query timed out after {timeout} seconds."
        
        except FileNotFoundError:
            logger.error(f"SQLcl not found: {self.sqlcl_path}")
            return False, f"SQLcl executable not found: {self.sqlcl_path}"
        
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return False, f"Execution error: {str(e)}"
        
        finally:
            # 임시 파일 정리
            if sql_file and os.path.exists(sql_file):
                try:
                    os.remove(sql_file)
                except Exception:
                    pass
    
    def test_connection(self):
        """연결 상태를 테스트합니다."""
        success, result = self.run_query("SELECT 'CONNECTION_OK' AS STATUS FROM DUAL", timeout=30)
        return success, result
    
    def is_connected(self):
        """연결 테스트 결과를 반환합니다."""
        return self._connection_tested

# Initialize Persistent Client
if 'sql_client' not in st.session_state:
    st.session_state.sql_client = SQLClient(SQLCL_PATH, DB_CONNECTION)

# --- Helper Functions ---

def display_data(df, show_chart=False, chart_type="bar"):
    """Displays data as a table or list based on row count, hiding empty columns.
    
    Args:
        df: DataFrame to display
        show_chart: Whether to show a chart
        chart_type: Type of chart ('bar', 'line', 'area', 'pie')
    """
    # Remove columns where all values are null/empty
    df_clean = df.dropna(axis=1, how='all')
    
    # 숫자 컬럼 소수점 1자리로 포맷팅
    for col in df_clean.select_dtypes(include=['float64', 'float32']).columns:
        df_clean[col] = df_clean[col].round(1)
    
    # 인덱스를 1부터 시작하도록 변경
    df_display = df_clean.reset_index(drop=True)
    df_display.index = df_display.index + 1
    df_display.index.name = "No"
    
    if len(df_clean) == 1 and not show_chart:
        # Single row display - Bullet points
        st.markdown("### 📋 상세 정보")
        row = df_clean.iloc[0]
        for col in df_clean.columns:
            val = row[col]
            # Skip if value is null or empty string
            if pd.isna(val) or str(val).strip() == "":
                continue
            # 숫자인 경우 소수점 1자리로 포맷
            if isinstance(val, float):
                st.markdown(f"- **{col}**: {val:.1f}")
            else:
                st.markdown(f"- **{col}**: {val}")
    else:
        # 차트 표시
        if show_chart and len(df_clean) > 0:
            display_chart(df_clean, chart_type)
        
        # Multiple rows - st.table 사용 (중앙 정렬 CSS 적용)
        # CSS 스타일 주입
        st.markdown("""
        <style>
            /* 테이블 전체 중앙 정렬 */
            div[data-testid="stTable"] table {
                width: 100%;
            }
            div[data-testid="stTable"] th {
                text-align: center !important;
                background-color: #f0f2f6;
                padding: 10px !important;
            }
            div[data-testid="stTable"] td {
                text-align: center !important;
                padding: 8px !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.table(df_display)


def display_chart(df, chart_type="bar"):
    """Display chart based on DataFrame.
    
    Automatically detects:
    - First text column as labels (x-axis)
    - First numeric column as values (y-axis)
    """
    # 데이터 복사 및 숫자 변환 시도
    df_chart = df.copy()
    
    # 모든 컬럼에 대해 숫자 변환 시도
    for col in df_chart.columns:
        try:
            df_chart[col] = pd.to_numeric(df_chart[col], errors='ignore')
        except Exception:
            pass
    
    # 컬럼 타입 분석
    text_cols = df_chart.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df_chart.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        st.warning("차트를 그릴 수치 데이터가 없습니다.")
        return
    
    # 라벨과 값 컬럼 자동 선택
    label_col = text_cols[0] if text_cols else df_chart.index.name or "Index"
    value_col = numeric_cols[0]
    
    # 차트용 데이터 준비
    if text_cols:
        chart_df = df_chart.set_index(text_cols[0])[[value_col]]
    else:
        chart_df = df_chart[[value_col]]
    
    st.markdown(f"### 📊 {value_col} by {label_col}")
    
    # 차트 타입별 렌더링
    if chart_type == "bar":
        st.bar_chart(chart_df, use_container_width=True)
    elif chart_type == "line":
        st.line_chart(chart_df, use_container_width=True)
    elif chart_type == "area":
        st.area_chart(chart_df, use_container_width=True)
    elif chart_type == "pie":
        # Streamlit doesn't have native pie chart, use plotly if available
        try:
            import plotly.express as px
            if text_cols:
                fig = px.pie(df_chart, names=text_cols[0], values=value_col, 
                            title=f"{value_col} by {label_col}")
            else:
                fig = px.pie(df_chart, values=value_col, title=f"{value_col} Distribution")
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.warning("파이 차트를 위해 plotly가 필요합니다. 'pip install plotly'로 설치해주세요.")
            st.bar_chart(chart_df, use_container_width=True)
        except Exception as e:
            st.warning(f"파이 차트 생성 오류: {e}")
            st.bar_chart(chart_df, use_container_width=True)
    else:
        st.bar_chart(chart_df, use_container_width=True)


def detect_chart_request(query):
    """Detect if user is requesting a chart/graph visualization.
    
    Returns:
        tuple: (is_chart_request: bool, chart_type: str)
    """
    query_lower = query.lower()
    
    # 차트/그래프 관련 키워드
    chart_keywords = ['차트', '그래프', 'chart', 'graph', '시각화', 'visualize', 'plot', '그려', '보여줘', '표시']
    
    # 차트 타입별 키워드
    bar_keywords = ['막대', 'bar', '바']
    line_keywords = ['라인', 'line', '선', '추이', '추세', 'trend']
    pie_keywords = ['파이', 'pie', '원형', '원그래프', '비율', '구성비', '도넛', 'donut']
    area_keywords = ['area', '영역', '면적']
    
    is_chart = any(kw in query_lower for kw in chart_keywords)
    
    # "원그래프"가 있으면 차트 요청으로 간주
    if '원그래프' in query_lower or '원 그래프' in query_lower:
        return True, "pie"
    
    if not is_chart:
        return False, "bar"
    
    # 차트 타입 감지
    if any(kw in query_lower for kw in pie_keywords):
        return True, "pie"
    elif any(kw in query_lower for kw in line_keywords):
        return True, "line"
    elif any(kw in query_lower for kw in area_keywords):
        return True, "area"
    else:
        return True, "bar"

def execute_sql_via_mcp(sql_query):
    """Executes SQL query via MCP Server."""
    if ClientSession is None:
        return False, "mcp package not installed. Run 'pip install mcp'."

    async def _run():
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "sqlcl_mcp.server"],
            env={
                "SQLCL_PATH": SQLCL_PATH,
                "DB_CONNECTION": DB_CONNECTION,
                "NLS_LANG": "KOREAN_KOREA.AL32UTF8",
                **os.environ
            }
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("execute_sql", arguments={"sql": sql_query})
                return result

    try:
        result = asyncio.run(_run())
        if not result.content:
            return False, "No content returned from MCP server"
            
        text = result.content[0].text
        if text.startswith("오류:"):
            return False, text
        return True, text
    except Exception as e:
        return False, f"MCP Execution Error: {str(e)}"

def execute_sql_direct(sql_query):
    """Executes SQL query using the persistent SQLClient."""
    if 'sql_client' in st.session_state:
        return st.session_state.sql_client.run_query(sql_query)
    else:
        return False, "SQL Client not initialized"

def execute_sql(sql_query):
    """Dispatches SQL execution based on selected mode."""
    mode = st.session_state.get("execution_mode", "Direct")
    if mode == "MCP Server (Standard)":
        return execute_sql_via_mcp(sql_query)
    else:
        return execute_sql_direct(sql_query)

@st.cache_data(ttl=3600)
def get_table_list():
    """Fetches table list from DB (Cached)."""
    sql = "SELECT table_name FROM user_tables ORDER BY table_name;"
    success, output = execute_sql(sql)
    if success:
        try:
            df = pd.read_csv(StringIO(output))
            if 'TABLE_NAME' in df.columns:
                return df['TABLE_NAME'].tolist()
            elif 'table_name' in df.columns:
                return df['table_name'].tolist()
            return []
        except Exception:
            return []
    return []

def clean_sql_response(content):
    """Extracts pure SQL from AI response."""
    # Remove markdown code blocks
    if "```" in content:
        match = re.search(r"```(?:sql)?\s*(.*?)```", content, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            content = content.replace("```sql", "").replace("```", "").strip()
    
    # Heuristic to remove non-SQL text
    upper_content = content.upper()
    valid_starts = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]
    if not any(upper_content.startswith(k) for k in valid_starts):
        match = re.search(r"(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s+.*", content, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(0)
            
    return content


def is_valid_sql(content):
    """Check if content is a valid SQL statement."""
    if not content:
        return False
    upper_content = content.strip().upper()
    valid_starts = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]
    return any(upper_content.startswith(k) for k in valid_starts)

def generate_sql_from_nl(nl_query, table_list, model_name, chat_history=None):
    """OpenAI 또는 Anthropic을 사용하여 자연어를 SQL로 변환합니다."""
    
    # 테이블 목록을 텍스트로 변환 (토큰 제한 고려하여 상위 100개만)
    tables_str = ", ".join(table_list[:100])
    if len(table_list) > 100:
        tables_str += f" 외 {len(table_list)-100}개"

    system_prompt = f"""
    You are an Oracle SQL expert.
    Convert the following natural language query into a valid Oracle SQL query.
    
    Context:
    - Database: Oracle
    - Available Tables: {tables_str}
    - Important Tables and Columns:
        - INMAST (Employee Master): 
            * EMPL_NUMB (Employee ID)
            * EMPL_NAME (Name)
            * DEPA_CODE (Dept Code)
            * IBSA_DATE (Entry/Join Date) - USE THIS FOR TENURE CALCULATION
            * SEX_GUBN (Gender)
            * BRTH_DATE (Birth Date)
            * EMPL_JKGB (Position/Rank Code)
        - ZME (Dept Master): DEPA_CODE, DEPA_NAME
        - HRM_PERSON: EMP_NO, SSN (Do NOT use for join - use INMAST directly)
    
    Rules:
    - Return ONLY the SQL query without any explanation or description.
    - NEVER include any text before or after the SQL query.
    - Use standard Oracle syntax.
    - ALWAYS use ENGLISH column aliases (e.g., DEPT_NAME, EMP_COUNT, AVG_TENURE). NEVER use Korean aliases.
    - ALWAYS use table aliases (e.g., INMAST I, ZME Z).
    - ALWAYS prefix column names with table alias.
    - For tenure/service years: ALWAYS use ROUND(MONTHS_BETWEEN(SYSDATE, I.IBSA_DATE) / 12, 1) to show 1 decimal place.
    - For any decimal/float results: ALWAYS use ROUND(..., 1) to limit to 1 decimal place.
    - Join INMAST and ZME: I.DEPA_CODE = Z.DEPA_CODE
    - Use I.EMPL_NUMB for counting employees.
    - Use FETCH FIRST n ROWS ONLY for limits.
    - For department name, use Z.DEPA_NAME from ZME table.
    - Do NOT join HRM_PERSON unless specifically needed for SSN or personal info.
    """
    
    # Claude 모델 사용
    if model_name.startswith("claude"):
        return _generate_sql_with_claude(nl_query, system_prompt, model_name, chat_history)
    else:
        return _generate_sql_with_openai(nl_query, system_prompt, model_name, chat_history)


def _generate_sql_with_claude(nl_query, system_prompt, model_name, chat_history=None):
    """Anthropic Claude를 사용하여 SQL 생성"""
    if not HAS_ANTHROPIC:
        return "-- anthropic 라이브러리가 설치되지 않았습니다.\n-- 터미널에서 'pip install anthropic' 명령어로 설치해주세요."
    
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "-- ANTHROPIC_API_KEY가 설정되지 않았습니다.\n-- .env 파일에 추가해주세요."
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # 메시지 구성
    messages = []
    
    if chat_history:
        for msg in chat_history:
            content = msg.get("content")
            if content is None:
                parts = []
                if "sql" in msg:
                    parts.append(f"Generated SQL: {msg['sql']}")
                if "error" in msg:
                    parts.append(f"Error: {msg['error']}")
                elif "data" in msg:
                    parts.append("Execution successful.")
                content = "\n".join(parts) if parts else "No content"
            
            messages.append({"role": msg["role"], "content": content})
    
    messages.append({"role": "user", "content": nl_query})
    
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=2048,
            system=system_prompt,
            messages=messages
        )
        content = response.content[0].text.strip()
        return clean_sql_response(content)
    except Exception as e:
        return f"-- Claude AI Error: {str(e)}"


def _generate_sql_with_openai(nl_query, system_prompt, model_name, chat_history=None):
    """OpenAI를 사용하여 SQL 생성"""
    if not HAS_OPENAI:
        return "-- OpenAI 라이브러리가 설치되지 않았습니다.\n-- 터미널에서 'pip install openai' 명령어로 설치해주세요."
    
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "-- OpenAI API Key가 입력되지 않았습니다.\n-- .env 파일을 확인해주세요."

    client = openai.OpenAI(api_key=api_key)
    
    messages = [{"role": "system", "content": system_prompt}]
    
    if chat_history:
        for msg in chat_history:
            content = msg.get("content")
            if content is None:
                parts = []
                if "sql" in msg:
                    parts.append(f"Generated SQL: {msg['sql']}")
                if "error" in msg:
                    parts.append(f"Error: {msg['error']}")
                elif "data" in msg:
                    parts.append("Execution successful.")
                content = "\n".join(parts) if parts else "No content"
            
            messages.append({"role": msg["role"], "content": content})
            
    messages.append({"role": "user", "content": nl_query})
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        content = response.choices[0].message.content.strip()
        return clean_sql_response(content)
    except Exception as e:
        return f"-- OpenAI Error: {str(e)}"

# --- Sidebar ---
with st.sidebar:
    # 헤더 영역
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <img src="https://img.icons8.com/color/96/oracle-logo.png" width="50">
        <h2 style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">SQLcl AI Explorer</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 연결 상태 카드
    if 'sql_client' in st.session_state:
        client = st.session_state.sql_client
        db_info = DB_CONNECTION.split('@')[1] if '@' in DB_CONNECTION else 'Unknown'
        
        if client.is_connected():
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%); 
                        padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.2rem;">🟢</span>
                    <span style="color: #90EE90; font-weight: 600;">연결됨</span>
                </div>
                <div style="color: #ccc; font-size: 0.8rem; margin-top: 0.5rem;">
                    📍 {db_info}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4a1a1a 0%, #5a2d2d 100%); 
                        padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.2rem;">🔴</span>
                    <span style="color: #FF6B6B; font-weight: 600;">연결 실패</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("🔄 연결 초기화", use_container_width=True):
        if 'sql_client' in st.session_state:
            del st.session_state.sql_client
        st.rerun()
    
    st.markdown("---")
    
    # AI 모델 선택 섹션
    st.markdown("##### 🤖 AI 모델")
    
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    
    model_options = [
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-5-20251101",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ]
    
    model_display = {
        "claude-sonnet-4-5-20250929": ("🟣", "Claude Sonnet 4.5", "균형잡힌 성능"),
        "claude-haiku-4-5-20251001": ("🟢", "Claude Haiku 4.5", "빠른 응답"),
        "claude-opus-4-5-20251101": ("🔵", "Claude Opus 4.5", "최고 성능"),
        "gpt-4o": ("🟡", "GPT-4o", "고성능"),
        "gpt-4-turbo": ("🟠", "GPT-4 Turbo", "안정적"),
        "gpt-3.5-turbo": ("⚪", "GPT-3.5 Turbo", "경제적"),
    }
    
    st.selectbox(
        "Model",
        model_options,
        index=0,
        key="selected_model",
        label_visibility="collapsed",
        format_func=lambda x: f"{model_display.get(x, ('', x, ''))[0]} {model_display.get(x, ('', x, ''))[1]}"
    )
    
    # 선택된 모델 정보 표시
    selected = st.session_state.selected_model
    if selected in model_display:
        icon, name, desc = model_display[selected]
        
        # API 상태 확인
        if selected.startswith("claude"):
            api_ready = HAS_ANTHROPIC and anthropic_api_key
        else:
            api_ready = HAS_OPENAI and openai_api_key
        
        status_color = "#90EE90" if api_ready else "#FFB347"
        status_text = "Ready" if api_ready else "API Key 필요"
        
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 0.8rem; border-radius: 8px; margin-top: 0.5rem;">
            <div style="color: #888; font-size: 0.75rem;">{desc}</div>
            <div style="display: flex; align-items: center; gap: 0.3rem; margin-top: 0.3rem;">
                <span style="width: 6px; height: 6px; background: {status_color}; border-radius: 50%;"></span>
                <span style="color: {status_color}; font-size: 0.7rem;">{status_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 차트 가이드
    with st.expander("📊 차트 사용법", expanded=False):
        st.markdown("""
        **키워드로 차트 요청:**
        - 🥧 `원그래프`, `파이` → 파이 차트
        - 📊 `막대`, `바` → 막대 차트  
        - 📈 `라인`, `추이` → 라인 차트
        - 📉 `영역` → 영역 차트
        
        **예시:**
        ```
        부서별 인원 원그래프로
        월별 매출 추이 보여줘
        ```
        """)
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.7rem;">
        Made with ❤️ by AI<br>
        <span style="font-size: 0.65rem;">v1.2.0 • 2024-12-02</span>
    </div>
    """, unsafe_allow_html=True)

# --- Main Interface ---
st.markdown("## 📊 SQLcl AI Explorer")
st.caption("자연어로 Oracle 데이터베이스를 탐색하세요")

# Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            if "sql" in msg:
                with st.expander("🔍 View SQL", expanded=False):
                    st.code(msg["sql"], language="sql")
            
            if "data" in msg:
                show_chart = msg.get("show_chart", False)
                chart_type = msg.get("chart_type", "bar")
                display_data(msg["data"], show_chart=show_chart, chart_type=chart_type)
            elif "error" in msg:
                st.error(msg["error"])
            elif "content" in msg and "sql" not in msg:
                st.write(msg["content"])
        else:
            st.write(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask a question (e.g., 부서별 인원수를 그래프로 보여줘)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 차트 요청 감지
    is_chart_request, chart_type = detect_chart_request(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating & Executing SQL..."):
            tables = get_table_list()
            history = st.session_state.messages[:-1]
            generated_sql = generate_sql_from_nl(prompt, tables, st.session_state.selected_model, history)
            
            with st.expander("🔍 View SQL", expanded=False):
                st.code(generated_sql, language="sql")
            
            # SQL 유효성 검사
            if not is_valid_sql(generated_sql):
                st.warning("AI가 유효한 SQL을 생성하지 못했습니다. 다시 시도해주세요.")
                st.code(generated_sql)
                message_data = {"role": "assistant", "content": "유효한 SQL을 생성하지 못했습니다.", "sql": generated_sql}
                st.session_state.messages.append(message_data)
                st.stop()
            
            success, output = execute_sql(generated_sql)
            message_data = {"role": "assistant", "sql": generated_sql}
            
            if success:
                try:
                    # CSV 파싱 전 데이터 정리
                    cleaned_output = output.strip()
                    if not cleaned_output:
                        st.warning("No data found.")
                        message_data["content"] = "No data found."
                    else:
                        csv_data = StringIO(cleaned_output)
                        # quotechar와 on_bad_lines 옵션으로 파싱 오류 방지
                        df = pd.read_csv(csv_data, quotechar='"', on_bad_lines='warn')
                        if df.empty:
                            st.warning("No data found.")
                            message_data["content"] = "No data found."
                        else:
                            display_data(df, show_chart=is_chart_request, chart_type=chart_type)
                            message_data["data"] = df
                            message_data["show_chart"] = is_chart_request
                            message_data["chart_type"] = chart_type
                except Exception as e:
                    st.warning(f"Failed to parse CSV: {e}")
                    st.code(output)
                    message_data["error"] = f"Parse Error: {output}"
            else:
                st.error("Execution Failed")
                st.code(output)
                message_data["error"] = f"Execution Failed: {output}"
            
            st.session_state.messages.append(message_data)
            st.session_state.sql_input = generated_sql

st.markdown("---")

# Manual SQL Area
with st.expander("📝 Manual SQL Execution", expanded=False):
    sql_input = st.text_area("SQL Query", value=st.session_state.sql_input, height=150)
    col1, col2 = st.columns([1, 6])
    if col1.button("▶️ Run"):
        with st.spinner("Running..."):
            success, output = execute_sql(sql_input)
            if success:
                try:
                    df = pd.read_csv(StringIO(output))
                    if df.empty:
                        st.warning("No data found.")
                    else:
                        display_data(df)
                except Exception:
                    st.code(output)
            else:
                st.error(output)
