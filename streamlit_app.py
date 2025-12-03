"""
SQLcl AI Explorer - 자연어로 Oracle DB 탐색

사용법:
    streamlit run streamlit_app.py
"""

import streamlit as st
import subprocess
import os
import sys
import asyncio
import pandas as pd
import re
import tempfile
import logging
from io import StringIO

from config import (
    SQLCL_PATH, DB_CONNECTION, SQLCL_INIT_COMMANDS,
    AI_MODELS, MODEL_DISPLAY, PAGE_CONFIG,
    DB_SCHEMA_INFO, SQL_GENERATION_RULES,
    CHART_KEYWORDS, LINE_CHART_KEYWORDS,
    PIE_CHART_KEYWORDS, AREA_CHART_KEYWORDS, APP_VERSION
)

# =============================================================================
# Optional Imports
# =============================================================================
try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
except ImportError:
    ClientSession = None
    stdio_client = None
    StdioServerParameters = None

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# =============================================================================
# Streamlit Configuration
# =============================================================================
st.set_page_config(**PAGE_CONFIG)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sqlcl-client")


# =============================================================================
# Session State
# =============================================================================
def init_session_state():
    """세션 상태 초기화"""
    defaults = {
        'history': [],
        'sql_input': "",
        'messages': [],
        'execution_mode': "Direct (Fast)",
        'selected_model': "claude-sonnet-4-5-20250929"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# =============================================================================
# SQL Client
# =============================================================================
class SQLClient:
    """파일 기반 SQLcl 클라이언트 - 안정적인 실행 보장"""
    
    def __init__(self, sqlcl_path: str, db_connection: str):
        self.sqlcl_path = sqlcl_path
        self.db_connection = db_connection
        self._connection_tested = False
        self._test_connection()
    
    def _get_env(self) -> dict:
        """SQLcl 실행 환경 변수"""
        env = os.environ.copy()
        env["NLS_LANG"] = "KOREAN_KOREA.AL32UTF8"
        env["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8"
        return env
    
    def _test_connection(self):
        """연결 테스트"""
        try:
            success, _ = self.run_query("SELECT 1 FROM DUAL")
            self._connection_tested = success
            logger.info(f"Connection test: {'OK' if success else 'Failed'}")
        except Exception as e:
            logger.error(f"Connection test error: {e}")
            self._connection_tested = False
    
    def run_query(self, query: str, timeout: int = 60) -> tuple[bool, str]:
        """SQL 쿼리 실행"""
        query = query.strip()
        if not query.endswith(";"):
            query += ";"
        
        sql_content = f"{SQLCL_INIT_COMMANDS}\n{query}\n\nEXIT;\n"
        sql_file = None
        
        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.sql', delete=False, encoding='utf-8'
            ) as f:
                f.write(sql_content)
                sql_file = f.name
            
            # SQLcl 실행
            result = subprocess.run(
                [self.sqlcl_path, "-S", self.db_connection, f"@{sql_file}"],
                capture_output=True, text=True, timeout=timeout,
                encoding='utf-8', errors='replace', env=self._get_env()
            )
            
            stdout = self._filter_output(result.stdout)
            stderr = self._filter_output(result.stderr)
            
            # 결과 파싱
            csv_lines, error_lines = self._parse_output(stdout)
            
            if error_lines and not csv_lines:
                return False, '\n'.join(error_lines)
            
            stdout = '\n'.join(csv_lines)
            
            if result.returncode != 0 and not stdout:
                return False, stderr or stdout or f"Exit code: {result.returncode}"
            
            if stdout.startswith("ORA-") or "SP2-" in stdout or "Error" in stdout[:50]:
                return False, stdout
            
            return True, stdout
            
        except subprocess.TimeoutExpired:
            return False, f"Query timeout ({timeout}s)"
        except FileNotFoundError:
            return False, f"SQLcl not found: {self.sqlcl_path}"
        except Exception as e:
            return False, f"Error: {str(e)}"
        finally:
            if sql_file and os.path.exists(sql_file):
                try:
                    os.remove(sql_file)
                except Exception:
                    pass
    
    def _filter_output(self, text: str) -> str:
        """JAVA_TOOL_OPTIONS 메시지 필터링"""
        if not text:
            return ""
        lines = [line for line in text.split('\n') if not line.startswith('Picked up JAVA_TOOL_OPTIONS')]
        return '\n'.join(lines).strip()
    
    def _parse_output(self, stdout: str) -> tuple[list, list]:
        """출력에서 CSV 데이터와 에러 분리"""
        csv_lines, error_lines = [], []
        in_error_block = False
        
        for line in stdout.split('\n'):
            stripped = line.strip()
            
            # 에러 감지
            if any(err in stripped for err in ['ORA-', 'SP2-', 'Error at']):
                in_error_block = True
                error_lines.append(line)
                continue
            
            # 에러 블록 내부
            if in_error_block:
                if any(kw in stripped for kw in ['*Cause:', '*Action:', 'https://docs.oracle']):
                    error_lines.append(line)
                    continue
                if line.startswith('       ') or line.startswith('\t'):
                    error_lines.append(line)
                    continue
                if not stripped:
                    continue
                in_error_block = False
            
            # 일반 데이터
            if stripped and stripped not in ['Execution successful.', 'Commit complete.']:
                csv_lines.append(line)
        
        return csv_lines, error_lines
    
    def is_connected(self) -> bool:
        return self._connection_tested


# SQL Client 초기화
if 'sql_client' not in st.session_state:
    st.session_state.sql_client = SQLClient(SQLCL_PATH, DB_CONNECTION)


# =============================================================================
# Display Functions
# =============================================================================
def display_data(df: pd.DataFrame, show_chart: bool = False, chart_type: str = "bar"):
    """데이터 표시 (테이블 + 차트)"""
    df_clean = df.dropna(axis=1, how='all')
    
    # 숫자 포맷팅
    for col in df_clean.select_dtypes(include=['float64', 'float32']).columns:
        df_clean[col] = df_clean[col].round(1)
    
    # 인덱스 1부터 시작
    df_display = df_clean.reset_index(drop=True)
    df_display.index = df_display.index + 1
    df_display.index.name = "No"
    
    # 단일 행: 상세 정보 표시
    if len(df_clean) == 1 and not show_chart:
        st.markdown("### 📋 상세 정보")
        for col in df_clean.columns:
            val = df_clean.iloc[0][col]
            if pd.notna(val) and str(val).strip():
                formatted = f"{val:.1f}" if isinstance(val, float) else val
                st.markdown(f"- **{col}**: {formatted}")
        return
    
    # 차트 표시
    if show_chart and len(df_clean) > 0:
        display_chart(df_clean, chart_type)
    
    # 테이블 스타일
    st.markdown("""
    <style>
        div[data-testid="stTable"] table { width: 100%; }
        div[data-testid="stTable"] th { text-align: center !important; background-color: #f0f2f6; padding: 10px !important; }
        div[data-testid="stTable"] td { text-align: center !important; padding: 8px !important; }
    </style>
    """, unsafe_allow_html=True)
    st.table(df_display)


def display_chart(df: pd.DataFrame, chart_type: str = "bar"):
    """차트 표시"""
    df_chart = df.copy()
    
    # 숫자 변환
    for col in df_chart.columns:
        df_chart[col] = pd.to_numeric(df_chart[col], errors='ignore')
    
    text_cols = df_chart.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df_chart.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        st.warning("차트를 그릴 수치 데이터가 없습니다.")
        return
    
    label_col = text_cols[0] if text_cols else "Index"
    value_col = numeric_cols[0]
    chart_df = df_chart.set_index(text_cols[0])[[value_col]] if text_cols else df_chart[[value_col]]
    
    st.markdown(f"### 📊 {value_col} by {label_col}")
    
    chart_funcs = {
        "bar": st.bar_chart,
        "line": st.line_chart,
        "area": st.area_chart
    }
    
    if chart_type in chart_funcs:
        chart_funcs[chart_type](chart_df, use_container_width=True)
    elif chart_type == "pie":
        try:
            import plotly.express as px
            fig = px.pie(df_chart, names=text_cols[0] if text_cols else None, values=value_col)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.warning("파이 차트를 위해 plotly 설치 필요: pip install plotly")
            st.bar_chart(chart_df, use_container_width=True)
    else:
        st.bar_chart(chart_df, use_container_width=True)


# =============================================================================
# Chart Detection
# =============================================================================
def detect_chart_request(query: str) -> tuple[bool, str]:
    """차트 요청 감지"""
    query_lower = query.lower()
    
    if '원그래프' in query_lower or '원 그래프' in query_lower:
        return True, "pie"
    
    if not any(kw in query_lower for kw in CHART_KEYWORDS):
        return False, "bar"
    
    for keywords, chart_type in [
        (PIE_CHART_KEYWORDS, "pie"),
        (LINE_CHART_KEYWORDS, "line"),
        (AREA_CHART_KEYWORDS, "area")
    ]:
        if any(kw in query_lower for kw in keywords):
            return True, chart_type
    
    return True, "bar"


# =============================================================================
# SQL Execution
# =============================================================================
def execute_sql_via_mcp(sql_query: str) -> tuple[bool, str]:
    """MCP 서버를 통한 SQL 실행"""
    if ClientSession is None:
        return False, "mcp package not installed"

    async def _run():
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "sqlcl_mcp.server"],
            env={"SQLCL_PATH": SQLCL_PATH, "DB_CONNECTION": DB_CONNECTION, 
                 "NLS_LANG": "KOREAN_KOREA.AL32UTF8", **os.environ}
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool("execute_sql", arguments={"sql": sql_query})

    try:
        result = asyncio.run(_run())
        if not result.content:
            return False, "No content returned"
        text = result.content[0].text
        return (False, text) if text.startswith("오류:") else (True, text)
    except Exception as e:
        return False, f"MCP Error: {str(e)}"


def execute_sql_direct(sql_query: str) -> tuple[bool, str]:
    """SQLClient를 통한 직접 실행"""
    if 'sql_client' in st.session_state:
        return st.session_state.sql_client.run_query(sql_query)
    return False, "SQL Client not initialized"


def execute_sql(sql_query: str) -> tuple[bool, str]:
    """실행 모드에 따른 SQL 실행"""
    mode = st.session_state.get("execution_mode", "Direct")
    if mode == "MCP Server (Standard)":
        return execute_sql_via_mcp(sql_query)
    return execute_sql_direct(sql_query)


@st.cache_data(ttl=3600)
def get_table_list() -> list:
    """테이블 목록 조회 (캐시됨)"""
    success, output = execute_sql("SELECT table_name FROM user_tables ORDER BY table_name")
    if success:
        try:
            df = pd.read_csv(StringIO(output))
            col = 'TABLE_NAME' if 'TABLE_NAME' in df.columns else 'table_name'
            return df[col].tolist() if col in df.columns else []
        except Exception:
            pass
    return []


# =============================================================================
# SQL Generation (AI)
# =============================================================================
def clean_sql_response(content: str) -> str:
    """AI 응답에서 SQL 추출"""
    if "```" in content:
        match = re.search(r"```(?:sql)?\s*(.*?)```", content, re.DOTALL)
        content = match.group(1).strip() if match else content.replace("```sql", "").replace("```", "").strip()
    
    upper = content.upper()
    valid_starts = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]
    
    if not any(upper.startswith(k) for k in valid_starts):
        match = re.search(r"(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s+.*", content, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(0)
    
    return content


def is_valid_sql(content: str) -> bool:
    """SQL 유효성 검사"""
    if not content:
        return False
    upper = content.strip().upper()
    return any(upper.startswith(k) for k in ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"])


def generate_sql_from_nl(nl_query: str, table_list: list, model_name: str, chat_history=None) -> str:
    """자연어 → SQL 변환"""
    tables_str = ", ".join(table_list[:100])
    if len(table_list) > 100:
        tables_str += f" 외 {len(table_list)-100}개"

    system_prompt = f"""You are an Oracle SQL expert.
Convert the natural language query into a valid Oracle SQL query.

Context:
- Database: Oracle
- Available Tables: {tables_str}
{DB_SCHEMA_INFO}
{SQL_GENERATION_RULES}
"""
    
    if model_name.startswith("claude"):
        return _generate_sql_claude(nl_query, system_prompt, model_name, chat_history)
    return _generate_sql_openai(nl_query, system_prompt, model_name, chat_history)


def _build_messages(chat_history: list, nl_query: str) -> list:
    """채팅 히스토리를 메시지로 변환"""
    messages = []
    if chat_history:
        for msg in chat_history:
            content = msg.get("content")
            if not content:
                parts = []
                if "sql" in msg:
                    parts.append(f"SQL: {msg['sql']}")
                if "error" in msg:
                    parts.append(f"Error: {msg['error']}")
                elif "data" in msg:
                    parts.append("OK")
                content = "\n".join(parts) or "No content"
            messages.append({"role": msg["role"], "content": content})
    messages.append({"role": "user", "content": nl_query})
    return messages


def _generate_sql_claude(nl_query: str, system_prompt: str, model_name: str, chat_history=None) -> str:
    """Claude로 SQL 생성"""
    if not HAS_ANTHROPIC:
        return "-- anthropic 라이브러리가 설치되지 않았습니다."
    
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "-- ANTHROPIC_API_KEY가 설정되지 않았습니다."
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        messages = _build_messages(chat_history, nl_query)
        response = client.messages.create(
            model=model_name, max_tokens=2048, system=system_prompt, messages=messages
        )
        return clean_sql_response(response.content[0].text.strip())
    except Exception as e:
        return f"-- Claude Error: {str(e)}"


def _generate_sql_openai(nl_query: str, system_prompt: str, model_name: str, chat_history=None) -> str:
    """OpenAI로 SQL 생성"""
    if not HAS_OPENAI:
        return "-- openai 라이브러리가 설치되지 않았습니다."
    
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "-- OPENAI_API_KEY가 설정되지 않았습니다."
    
    try:
        client = openai.OpenAI(api_key=api_key)
        messages = [{"role": "system", "content": system_prompt}] + _build_messages(chat_history, nl_query)
        response = client.chat.completions.create(model=model_name, messages=messages)
        return clean_sql_response(response.choices[0].message.content.strip())
    except Exception as e:
        return f"-- OpenAI Error: {str(e)}"


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    # 헤더
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <img src="https://img.icons8.com/color/96/oracle-logo.png" width="50">
        <h2 style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">SQLcl AI Explorer</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 연결 상태
    if 'sql_client' in st.session_state:
        client = st.session_state.sql_client
        db_info = DB_CONNECTION.split('@')[1] if '@' in DB_CONNECTION else 'Unknown'
        
        if client.is_connected():
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a472a, #2d5a3d); padding: 1rem; border-radius: 10px;">
                <span style="font-size: 1.2rem;">🟢</span>
                <span style="color: #90EE90; font-weight: 600;">연결됨</span>
                <div style="color: #ccc; font-size: 0.8rem; margin-top: 0.5rem;">📍 {db_info}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4a1a1a, #5a2d2d); padding: 1rem; border-radius: 10px;">
                <span style="font-size: 1.2rem;">🔴</span>
                <span style="color: #FF6B6B; font-weight: 600;">연결 실패</span>
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("🔄 연결 초기화", use_container_width=True):
        if 'sql_client' in st.session_state:
            del st.session_state.sql_client
        st.rerun()
    
    st.markdown("---")
    
    # AI 모델 선택
    st.markdown("##### 🤖 AI 모델")
    
    st.selectbox(
        "Model", AI_MODELS, index=0, key="selected_model",
        label_visibility="collapsed",
        format_func=lambda x: f"{MODEL_DISPLAY.get(x, ('', x, ''))[0]} {MODEL_DISPLAY.get(x, ('', x, ''))[1]}"
    )
    
    # 모델 상태
    selected = st.session_state.selected_model
    if selected in MODEL_DISPLAY:
        _, _, desc = MODEL_DISPLAY[selected]
        api_key = os.getenv("ANTHROPIC_API_KEY" if selected.startswith("claude") else "OPENAI_API_KEY", "")
        has_lib = HAS_ANTHROPIC if selected.startswith("claude") else HAS_OPENAI
        api_ready = has_lib and api_key
        
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 0.8rem; border-radius: 8px; margin-top: 0.5rem;">
            <div style="color: #888; font-size: 0.75rem;">{desc}</div>
            <div style="display: flex; align-items: center; gap: 0.3rem; margin-top: 0.3rem;">
                <span style="width: 6px; height: 6px; background: {'#90EE90' if api_ready else '#FFB347'}; border-radius: 50%;"></span>
                <span style="color: {'#90EE90' if api_ready else '#FFB347'}; font-size: 0.7rem;">{'Ready' if api_ready else 'API Key 필요'}</span>
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
        """)
    
    # 푸터
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.7rem;">
        Made with ❤️ by AI<br>
        <span style="font-size: 0.65rem;">{APP_VERSION}</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Main Interface
# =============================================================================
st.markdown("## 📊 SQLcl AI Explorer")
st.caption("자연어로 Oracle 데이터베이스를 탐색하세요")

# 채팅 히스토리 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            if "sql" in msg:
                with st.expander("🔍 View SQL", expanded=False):
                    st.code(msg["sql"], language="sql")
            
            if "data" in msg:
                display_data(msg["data"], msg.get("show_chart", False), msg.get("chart_type", "bar"))
            elif "error" in msg:
                st.error(msg["error"])
            elif "content" in msg and "sql" not in msg:
                st.write(msg["content"])
        else:
            st.write(msg["content"])

# 채팅 입력
if prompt := st.chat_input("질문하세요 (예: 부서별 인원수를 그래프로 보여줘)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    is_chart_request, chart_type = detect_chart_request(prompt)

    with st.chat_message("assistant"):
        with st.spinner("SQL 생성 및 실행 중..."):
            tables = get_table_list()
            history = st.session_state.messages[:-1]
            generated_sql = generate_sql_from_nl(prompt, tables, st.session_state.selected_model, history)
            
            with st.expander("🔍 View SQL", expanded=False):
                st.code(generated_sql, language="sql")
            
            if not is_valid_sql(generated_sql):
                st.warning("유효한 SQL을 생성하지 못했습니다.")
                st.code(generated_sql)
                st.session_state.messages.append({"role": "assistant", "content": "SQL 생성 실패", "sql": generated_sql})
                st.stop()
            
            success, output = execute_sql(generated_sql)
            message_data = {"role": "assistant", "sql": generated_sql}
            
            if success:
                try:
                    if output.strip():
                        df = pd.read_csv(StringIO(output), quotechar='"', on_bad_lines='warn')
                        if not df.empty:
                            display_data(df, is_chart_request, chart_type)
                            message_data.update({"data": df, "show_chart": is_chart_request, "chart_type": chart_type})
                        else:
                            st.warning("데이터 없음")
                            message_data["content"] = "No data"
                    else:
                        st.warning("데이터 없음")
                        message_data["content"] = "No data"
                except Exception as e:
                    st.warning(f"파싱 오류: {e}")
                    st.code(output)
                    message_data["error"] = f"Parse Error: {output}"
            else:
                st.error("실행 실패")
                st.code(output)
                message_data["error"] = f"Execution Failed: {output}"
            
            st.session_state.messages.append(message_data)
            st.session_state.sql_input = generated_sql

st.markdown("---")

# 수동 SQL 실행
with st.expander("📝 수동 SQL 실행", expanded=False):
    sql_input = st.text_area("SQL Query", value=st.session_state.sql_input, height=150)
    if st.button("▶️ 실행"):
        with st.spinner("실행 중..."):
            success, output = execute_sql(sql_input)
            if success:
                try:
                    df = pd.read_csv(StringIO(output))
                    display_data(df) if not df.empty else st.warning("데이터 없음")
                except Exception:
                    st.code(output)
            else:
                st.error(output)
