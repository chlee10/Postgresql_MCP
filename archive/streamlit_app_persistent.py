"""
SQLcl AI Explorer with Persistent MCP Server

이 앱은 Persistent DB 연결을 사용하여 빠른 쿼리 실행을 제공합니다.
- MCP 서버는 백그라운드에서 실행되며 DB 연결을 유지합니다
- 매 쿼리마다 로그인하지 않아 빠른 응답 (0.1~0.5초)
"""

import streamlit as st
import os
import sys
import asyncio
import pandas as pd
import re
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
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
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
    page_title="SQLcl AI Explorer (Persistent)",
    page_icon="⚡",
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
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "claude-sonnet-4-5-20250929"
if "mcp_session" not in st.session_state:
    st.session_state.mcp_session = None
if "query_times" not in st.session_state:
    st.session_state.query_times = []

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sqlcl-client")


# --- Persistent MCP Client ---
class PersistentMCPClient:
    """
    MCP 서버와의 Persistent 연결을 관리하는 클라이언트.
    
    서버 프로세스를 백그라운드에서 유지하고, 쿼리 시 재사용합니다.
    """
    
    def __init__(self, sqlcl_path: str, db_connection: str):
        self.sqlcl_path = sqlcl_path
        self.db_connection = db_connection
        self.process = None
        self.session = None
        self.read_stream = None
        self.write_stream = None
        self._connected = False
        self._connection_time = None
        
    async def connect(self):
        """MCP 서버에 연결합니다."""
        if self._connected and self.process and self.process.returncode is None:
            return True
        
        try:
            logger.info("Starting Persistent MCP Server...")
            
            server_params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "sqlcl_mcp.persistent_server"],
                env={
                    "SQLCL_PATH": self.sqlcl_path,
                    "DB_CONNECTION": self.db_connection,
                    "NLS_LANG": "KOREAN_KOREA.AL32UTF8",
                    **os.environ
                }
            )
            
            # stdio_client를 사용하여 연결
            self._client_cm = stdio_client(server_params)
            self.read_stream, self.write_stream = await self._client_cm.__aenter__()
            
            self._session_cm = ClientSession(self.read_stream, self.write_stream)
            self.session = await self._session_cm.__aenter__()
            
            await self.session.initialize()
            
            self._connected = True
            self._connection_time = time.time()
            logger.info("✅ MCP Server connected!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MCP Server: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """MCP 서버 연결을 종료합니다."""
        try:
            if self.session:
                await self._session_cm.__aexit__(None, None, None)
            if self._client_cm:
                await self._client_cm.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"Disconnect warning: {e}")
        finally:
            self._connected = False
            self.session = None
            logger.info("MCP Server disconnected")
    
    async def execute_sql(self, sql: str, timeout: float = 60.0) -> tuple[bool, str]:
        """SQL 쿼리를 실행합니다."""
        if not self._connected:
            connected = await self.connect()
            if not connected:
                return False, "Failed to connect to MCP server"
        
        try:
            result = await self.session.call_tool(
                "execute_sql",
                arguments={"sql": sql, "timeout": timeout}
            )
            
            if not result.content:
                return False, "No content returned from MCP server"
            
            text = result.content[0].text
            if text.startswith("오류:"):
                return False, text
            
            return True, text
            
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            self._connected = False  # 연결 상태 리셋
            return False, f"Execution error: {str(e)}"
    
    async def check_status(self) -> tuple[bool, str]:
        """연결 상태를 확인합니다."""
        if not self._connected:
            return False, "Not connected"
        
        try:
            result = await self.session.call_tool("connection_status", arguments={})
            return True, result.content[0].text
        except Exception as e:
            return False, str(e)
    
    def is_connected(self) -> bool:
        return self._connected
    
    def get_uptime(self) -> float:
        """연결 유지 시간 (초)"""
        if self._connection_time:
            return time.time() - self._connection_time
        return 0


# 동기 래퍼 함수들
def get_or_create_mcp_client():
    """MCP 클라이언트를 가져오거나 생성합니다."""
    if 'mcp_client' not in st.session_state or st.session_state.mcp_client is None:
        st.session_state.mcp_client = PersistentMCPClient(SQLCL_PATH, DB_CONNECTION)
    return st.session_state.mcp_client


def execute_sql_persistent(sql_query: str) -> tuple[bool, str, float]:
    """Persistent MCP 서버를 통해 SQL을 실행합니다."""
    if not HAS_MCP:
        return False, "MCP package not installed", 0
    
    client = get_or_create_mcp_client()
    
    async def _run():
        start = time.time()
        if not client.is_connected():
            await client.connect()
        success, result = await client.execute_sql(sql_query)
        elapsed = time.time() - start
        return success, result, elapsed
    
    try:
        return asyncio.run(_run())
    except Exception as e:
        return False, f"Error: {str(e)}", 0


def check_connection_status() -> tuple[bool, str]:
    """연결 상태를 확인합니다."""
    if not HAS_MCP:
        return False, "MCP not installed"
    
    client = get_or_create_mcp_client()
    
    async def _run():
        return await client.check_status()
    
    try:
        return asyncio.run(_run())
    except Exception as e:
        return False, str(e)


# --- Helper Functions ---

def display_data(df, show_chart=False, chart_type="bar"):
    """Displays data as a table or list based on row count."""
    df_clean = df.dropna(axis=1, how='all')
    
    for col in df_clean.select_dtypes(include=['float64', 'float32']).columns:
        df_clean[col] = df_clean[col].round(1)
    
    df_display = df_clean.reset_index(drop=True)
    df_display.index = df_display.index + 1
    df_display.index.name = "No"
    
    if len(df_clean) == 1 and not show_chart:
        st.markdown("### 📋 상세 정보")
        row = df_clean.iloc[0]
        for col in df_clean.columns:
            val = row[col]
            if pd.isna(val) or str(val).strip() == "":
                continue
            if isinstance(val, float):
                st.markdown(f"- **{col}**: {val:.1f}")
            else:
                st.markdown(f"- **{col}**: {val}")
    else:
        if show_chart and len(df_clean) > 0:
            display_chart(df_clean, chart_type)
        
        st.markdown("""
        <style>
            div[data-testid="stTable"] table { width: 100%; }
            div[data-testid="stTable"] th { text-align: center !important; background-color: #f0f2f6; padding: 10px !important; }
            div[data-testid="stTable"] td { text-align: center !important; padding: 8px !important; }
        </style>
        """, unsafe_allow_html=True)
        
        st.table(df_display)


def display_chart(df, chart_type="bar"):
    """Display chart based on DataFrame."""
    df_chart = df.copy()
    
    for col in df_chart.columns:
        try:
            df_chart[col] = pd.to_numeric(df_chart[col], errors='ignore')
        except Exception:
            pass
    
    text_cols = df_chart.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df_chart.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        st.warning("차트를 그릴 수치 데이터가 없습니다.")
        return
    
    label_col = text_cols[0] if text_cols else df_chart.index.name or "Index"
    value_col = numeric_cols[0]
    
    if text_cols:
        chart_df = df_chart.set_index(text_cols[0])[[value_col]]
    else:
        chart_df = df_chart[[value_col]]
    
    st.markdown(f"### 📊 {value_col} by {label_col}")
    
    if chart_type == "bar":
        st.bar_chart(chart_df, use_container_width=True)
    elif chart_type == "line":
        st.line_chart(chart_df, use_container_width=True)
    elif chart_type == "area":
        st.area_chart(chart_df, use_container_width=True)
    elif chart_type == "pie":
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
            st.warning("파이 차트를 위해 plotly가 필요합니다.")
            st.bar_chart(chart_df, use_container_width=True)
    else:
        st.bar_chart(chart_df, use_container_width=True)


def detect_chart_request(query):
    """Detect if user is requesting a chart/graph visualization."""
    query_lower = query.lower()
    
    chart_keywords = ['차트', '그래프', 'chart', 'graph', '시각화', 'visualize', 'plot', '그려', '보여줘', '표시']
    line_keywords = ['라인', 'line', '선', '추이', '추세', 'trend']
    pie_keywords = ['파이', 'pie', '원형', '원그래프', '비율', '구성비', '도넛', 'donut']
    area_keywords = ['area', '영역', '면적']
    
    if '원그래프' in query_lower or '원 그래프' in query_lower:
        return True, "pie"
    
    is_chart = any(kw in query_lower for kw in chart_keywords)
    
    if not is_chart:
        return False, "bar"
    
    if any(kw in query_lower for kw in pie_keywords):
        return True, "pie"
    elif any(kw in query_lower for kw in line_keywords):
        return True, "line"
    elif any(kw in query_lower for kw in area_keywords):
        return True, "area"
    else:
        return True, "bar"


@st.cache_data(ttl=3600)
def get_table_list():
    """Fetches table list from DB (Cached)."""
    sql = "SELECT table_name FROM user_tables ORDER BY table_name"
    success, output, _ = execute_sql_persistent(sql)
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
    if "```" in content:
        match = re.search(r"```(?:sql)?\s*(.*?)```", content, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            content = content.replace("```sql", "").replace("```", "").strip()
    
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
    """자연어를 SQL로 변환합니다."""
    
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
    
    if model_name.startswith("claude"):
        return _generate_sql_with_claude(nl_query, system_prompt, model_name, chat_history)
    else:
        return _generate_sql_with_openai(nl_query, system_prompt, model_name, chat_history)


def _generate_sql_with_claude(nl_query, system_prompt, model_name, chat_history=None):
    """Anthropic Claude를 사용하여 SQL 생성"""
    if not HAS_ANTHROPIC:
        return "-- anthropic 라이브러리가 설치되지 않았습니다."
    
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "-- ANTHROPIC_API_KEY가 설정되지 않았습니다."
    
    client = anthropic.Anthropic(api_key=api_key)
    
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
        return "-- OpenAI 라이브러리가 설치되지 않았습니다."
    
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "-- OpenAI API Key가 입력되지 않았습니다."

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
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 3rem;">⚡</span>
        <h2 style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">SQLcl AI Explorer</h2>
        <p style="color: #888; font-size: 0.8rem; margin: 0;">Persistent Connection Mode</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 연결 상태 확인
    client = get_or_create_mcp_client() if HAS_MCP else None
    
    if client and client.is_connected():
        uptime = client.get_uptime()
        uptime_str = f"{int(uptime // 60)}분 {int(uptime % 60)}초" if uptime >= 60 else f"{uptime:.0f}초"
        db_info = DB_CONNECTION.split('@')[1] if '@' in DB_CONNECTION else 'Unknown'
        
        # 평균 쿼리 시간
        avg_time = sum(st.session_state.query_times[-10:]) / len(st.session_state.query_times[-10:]) if st.session_state.query_times else 0
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%); 
                    padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.2rem;">⚡</span>
                <span style="color: #90EE90; font-weight: 600;">Persistent 연결</span>
            </div>
            <div style="color: #ccc; font-size: 0.8rem; margin-top: 0.5rem;">
                📍 {db_info}<br>
                ⏱️ 연결 유지: {uptime_str}<br>
                🚀 평균 응답: {avg_time:.2f}초
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4a3a1a 0%, #5a4d2d 100%); 
                    padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.2rem;">🟡</span>
                <span style="color: #FFD700; font-weight: 600;">대기 중</span>
            </div>
            <div style="color: #ccc; font-size: 0.8rem; margin-top: 0.3rem;">
                첫 쿼리 시 자동 연결됩니다
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    if col1.button("🔄 재연결", use_container_width=True):
        if 'mcp_client' in st.session_state:
            st.session_state.mcp_client = None
        st.cache_data.clear()
        st.rerun()
    
    if col2.button("🧹 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_times = []
        st.rerun()
    
    st.markdown("---")
    
    # AI 모델 선택
    st.markdown("##### 🤖 AI 모델")
    
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
    
    st.markdown("---")
    
    # 성능 비교
    with st.expander("📈 성능 비교", expanded=False):
        st.markdown("""
        | 항목 | 기존 방식 | Persistent |
        |------|----------|------------|
        | 첫 쿼리 | ~5초 | ~5초 |
        | 이후 쿼리 | ~5초 | **~0.2초** |
        | 10개 쿼리 | ~50초 | **~7초** |
        
        **왜 빠른가요?**
        - DB 로그인을 1회만 수행
        - SQLcl 프로세스 재사용
        - 연결 오버헤드 제거
        """)
    
    # 차트 가이드
    with st.expander("📊 차트 사용법", expanded=False):
        st.markdown("""
        **키워드로 차트 요청:**
        - 🥧 `원그래프`, `파이` → 파이 차트
        - 📊 `막대`, `바` → 막대 차트  
        - 📈 `라인`, `추이` → 라인 차트
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.7rem;">
        Made with ❤️ by AI<br>
        <span style="font-size: 0.65rem;">v2.0.0 • Persistent Mode</span>
    </div>
    """, unsafe_allow_html=True)


# --- Main Interface ---
st.markdown("## ⚡ SQLcl AI Explorer")
st.caption("Persistent 연결로 빠른 Oracle 데이터베이스 탐색")

# Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            if "sql" in msg:
                with st.expander("🔍 View SQL", expanded=False):
                    st.code(msg["sql"], language="sql")
            
            if "elapsed" in msg:
                st.caption(f"⚡ 실행 시간: {msg['elapsed']:.2f}초")
            
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

    is_chart_request, chart_type = detect_chart_request(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating SQL..."):
            tables = get_table_list()
            history = st.session_state.messages[:-1]
            generated_sql = generate_sql_from_nl(prompt, tables, st.session_state.selected_model, history)
            
            with st.expander("🔍 View SQL", expanded=False):
                st.code(generated_sql, language="sql")
            
            if not is_valid_sql(generated_sql):
                st.warning("AI가 유효한 SQL을 생성하지 못했습니다.")
                st.code(generated_sql)
                message_data = {"role": "assistant", "content": "유효한 SQL을 생성하지 못했습니다.", "sql": generated_sql}
                st.session_state.messages.append(message_data)
                st.stop()
            
            with st.spinner("Executing SQL... ⚡"):
                success, output, elapsed = execute_sql_persistent(generated_sql)
            
            # 실행 시간 기록
            st.session_state.query_times.append(elapsed)
            st.caption(f"⚡ 실행 시간: {elapsed:.2f}초")
            
            message_data = {"role": "assistant", "sql": generated_sql, "elapsed": elapsed}
            
            if success:
                try:
                    cleaned_output = output.strip()
                    if not cleaned_output:
                        st.warning("No data found.")
                        message_data["content"] = "No data found."
                    else:
                        csv_data = StringIO(cleaned_output)
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
            success, output, elapsed = execute_sql_persistent(sql_input)
            st.session_state.query_times.append(elapsed)
            st.caption(f"⚡ 실행 시간: {elapsed:.2f}초")
            
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
