"""
PostgreSQL AI Explorer - 자연어로 PostgreSQL DB 탐색 (MCP SSE 기반)

사용법:
    1. MCP 서버 시작: poetry run python -m postgresql_mcp.server --sse
    2. Streamlit 앱: streamlit run streamlit_app.py
"""

import streamlit as st
import os
import asyncio
import pandas as pd
import re
import logging
import httpx
import json
from io import StringIO
from typing import List, Dict, Any, Tuple, Optional

from config import (
    SERVER_URL,
    AI_MODELS, MODEL_DISPLAY, PAGE_CONFIG, DEFAULT_MODEL,
    DB_SCHEMA_INFO, SQL_GENERATION_RULES,
    CHART_KEYWORDS, LINE_CHART_KEYWORDS,
    PIE_CHART_KEYWORDS, AREA_CHART_KEYWORDS, APP_VERSION
)

# =============================================================================
# MCP SSE Client
# =============================================================================
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    HAS_MCP = True
except ImportError:
    ClientSession = None
    sse_client = None
    HAS_MCP = False

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
logger = logging.getLogger("postgresql-client")


# =============================================================================
# Session State
# =============================================================================
def init_session_state():
    """세션 상태 초기화"""
    defaults = {
        'history': [],
        'sql_input': "",
        'messages': [],
        'selected_model': DEFAULT_MODEL
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================================================================
# Helper Functions
# =============================================================================

def parse_result(text_content: str) -> List[Dict[str, Any]]:
    """Parse JSON result from MCP server"""
    try:
        return json.loads(text_content)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON: {text_content}")
        return []

# =============================================================================
# Display Functions
# =============================================================================
def display_data(df: pd.DataFrame, show_chart: bool = False, chart_type: str = "bar", query: str = ""):
    """데이터 표시 (테이블 + 차트)"""
    if df.empty:
        st.info("데이터가 없습니다.")
        return

    # 숫자 포맷팅 - float를 소수점 1자리 문자열로 변환 (optional)
    # Pandas display handles this well, maybe simpler is better
    
    # 인덱스 1부터 시작
    df_display = df.reset_index(drop=True)
    df_display.index = df_display.index + 1
    df_display.index.name = "No"
    
    # 차트 표시
    if show_chart and len(df) > 0:
        display_chart(df, chart_type, query)
    else:
        # 표만 나올 때 제목 추가
        table_title = extract_chart_title(query) if query else "조회 결과"
        st.markdown(f"#### 📋 {table_title}")
    
    st.table(df_display.head(100)) # Show first 100 rows to avoid UI lag


def display_chart(df: pd.DataFrame, chart_type: str = "bar", query: str = ""):
    """차트 표시"""
    import uuid
    
    df_chart = df.copy()
    
    # 데이터가 너무 많으면 차트용으로 자름
    if len(df_chart) > 50:
         st.warning("데이터가 많아 상위 50개만 차트에 표시합니다.")
         df_chart = df_chart.head(50)

    # 컬럼 타입 확인
    # 첫 번째 컬럼: Label
    # 나머지 숫자 컬럼: Values
    
    if len(df_chart.columns) < 2:
        st.warning("차트를 그리려면 최소 2개의 컬럼이 필요합니다.")
        return

    # 자동 타입 변환 시도
    df_chart = df_chart.convert_dtypes()
    
    label_col = df_chart.columns[0]
    
    # 숫자형 컬럼 찾기 (첫번째 컬럼 제외)
    numeric_cols = []
    for col in df_chart.columns[1:]:
        if pd.api.types.is_numeric_dtype(df_chart[col]):
            numeric_cols.append(col)
            
    if not numeric_cols:
        # 강제 변환 시도
        for col in df_chart.columns[1:]:
            try:
                df_chart[col] = pd.to_numeric(df_chart[col])
                numeric_cols.append(col)
            except:
                pass

    if not numeric_cols:
        st.warning("차트를 그릴 수치 데이터가 없습니다.")
        return
    
    value_col = numeric_cols[0] # 첫 번째 수치 컬럼 사용
    
    # 쿼리에서 차트 제목 추출
    chart_title = extract_chart_title(query) if query else f"{value_col} by {label_col}"
    st.markdown(f"#### 📊 {chart_title}")
    
    # Plotly로 모든 차트 그리기
    try:
        import plotly.express as px
        
        if chart_type == "pie":
            fig = px.pie(df_chart, names=label_col, values=value_col)
            fig.update_traces(textposition='inside', textinfo='percent+label')
        elif chart_type == "line":
            fig = px.line(df_chart, x=label_col, y=value_col, markers=True)
            fig.update_layout(xaxis_title=label_col, yaxis_title=value_col)
        elif chart_type == "area":
            fig = px.area(df_chart, x=label_col, y=value_col)
            fig.update_layout(xaxis_title=label_col, yaxis_title=value_col)
        else:  # bar (default)
            fig = px.bar(df_chart, x=label_col, y=value_col)
            fig.update_layout(xaxis_title=label_col, yaxis_title=value_col)
            fig.update_xaxes(type='category') # Ensure sequential order
        
        chart_key = f"chart_{uuid.uuid4().hex[:8]}"
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
    except ImportError:
        # Fallback
        chart_df = df_chart.set_index(label_col)[[value_col]]
        if chart_type == "line":
            st.line_chart(chart_df)
        elif chart_type == "area":
            st.area_chart(chart_df)
        else:
            st.bar_chart(chart_df)


def extract_chart_title(query: str) -> str:
    """쿼리에서 차트 제목 추출"""
    import re
    title = query.strip()
    title = re.split(r'[,.]', title)[0].strip()
    # Simple replacement for demonstration
    return title


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
# SQL Execution (SSE MCP Client)
# =============================================================================
def execute_tool_mcp(tool_name: str, arguments: dict) -> Tuple[bool, Any]:
    """Execute generic tool via MCP"""
    if not HAS_MCP:
        return False, "mcp package not installed"

    async def _run():
        sse_url = f"{SERVER_URL}/sse"
        try:
            async with sse_client(sse_url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool(tool_name, arguments=arguments)
        except Exception as e:
            raise e

    try:
        result = asyncio.run(_run())
        if not result.content:
            return False, "No content returned"
        
        text = result.content[0].text
        
        # Parse JSON
        parsed = parse_result(text)
        
        # Check for error dict
        if isinstance(parsed, dict) and "error" in parsed:
            return False, parsed["error"]
        
        return True, parsed
        
    except httpx.ConnectError:
        return False, "MCP 서버에 연결할 수 없습니다. 서버를 먼저 시작하세요."
    except Exception as e:
        return False, f"MCP Error: {str(e)}"

@st.cache_data(ttl=3600)
def get_table_list() -> list:
    """테이블 목록 조회 (캐시됨)"""
    success, data = execute_tool_mcp("list_tables", {})
    if success and isinstance(data, list):
        # Result format: [{'table_schema': '..', 'table_name': '..'}]
        return [row['table_name'] for row in data if 'table_name' in row]
    return []

# =============================================================================
# SQL Generation (AI)
# =============================================================================
def clean_sql_response(content: str) -> str:
    """AI 응답에서 SQL 추출"""
    # 1. Look for markdown code blocks first
    if "```" in content:
        match = re.search(r"```(?:sql)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # 2. explicit "SQL:" prefix
    content = re.sub(r'^SQL:\s*', '', content, flags=re.IGNORECASE).strip()
    
    # 3. If content starts with "Here is the SQL" or similar, try to find the actual SQL
    # Simple heuristic: find the first SELECT, INSERT, UPDATE, DELETE, WITH
    match = re.search(r'(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|ALTER|DROP)\s+.*', content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()
            
    return content.strip()

def is_valid_sql(content: str) -> bool:
    """SQL 유효성 검사"""
    if not content: return False
    return True # Allow permissive for now

def generate_sql_from_nl(nl_query: str, table_list: list, model_name: str, chat_history=None) -> str:
    """자연어 → SQL 변환"""
    tables_str = ", ".join(table_list[:100])
    if len(table_list) > 100:
        tables_str += f" 외 {len(table_list)-100}개"

    system_prompt = f"""You are a PostgreSQL expert.
Convert the natural language query into a valid PostgreSQL SQL query.

Context:
- Database: PostgreSQL
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
            content = msg.get("content", "")
            if "sql" in msg:
                 content += f"\nSQL: {msg['sql']}"
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
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 2.5rem;">🐘</span>
        <h2 style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">PostgreSQL MCP</h2>
        <p style="color: #888; font-size: 0.75rem; margin: 0.3rem 0 0 0;">Explorer</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # AI 모델 선택
    st.markdown("##### 🤖 AI Model")
    st.selectbox(
        "Model", AI_MODELS, key="selected_model",
        label_visibility="collapsed",
        format_func=lambda x: MODEL_DISPLAY.get(x, x)
    )
    
    st.markdown("---")

    # Table List
    st.markdown("##### 📂 Tables")
    with st.spinner("Loading tables..."):
        tables = get_table_list()
    
    if tables:
        st.markdown(f"Found **{len(tables)}** tables")
        with st.expander("View Tables", expanded=False):
            st.markdown("\n".join([f"- {t}" for t in tables]))
    else:
        st.info("No tables found in 'public' schema.")
        st.caption("Check connection settings in .env")

    st.markdown("---")
    
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.7rem;">
        MCP + PostgreSQL<br>
        <span style="font-size: 0.65rem;">v{APP_VERSION}</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Main Interface
# =============================================================================
st.markdown("## 📊 PostgreSQL AI Explorer")
st.caption("자연어로 PostgreSQL 데이터베이스를 탐색하세요")

# 채팅 히스토리 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            if "sql" in msg:
                with st.expander("🔍 View SQL", expanded=False):
                    st.code(msg["sql"], language="sql")
            
            if "data" in msg:
                display_data(msg["data"], msg.get("show_chart", False), msg.get("chart_type", "bar"), msg.get("query", ""))
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
        result_placeholder = st.empty()
        
        with result_placeholder.container():
            with st.spinner("SQL 생성 및 실행 중..."):
                tables = get_table_list()
                history = st.session_state.messages[:-1]
                generated_sql = generate_sql_from_nl(prompt, tables, st.session_state.selected_model, history)
        
        result_placeholder.empty()
        
        with st.expander("🔍 View SQL", expanded=False):
            st.code(generated_sql, language="sql")
        
        if generated_sql.startswith("--"):
             st.error(generated_sql)
             st.session_state.messages.append({"role": "assistant", "content": generated_sql})
             st.stop()

        with st.spinner("쿼리 실행 중..."):
            success, result_data = execute_tool_mcp("query", {"sql": generated_sql})
        
        message_data = {"role": "assistant", "sql": generated_sql}
        
        if success:
             # result_data is List[Dict]
             if result_data:
                 df = pd.DataFrame(result_data)
                 display_data(df, is_chart_request, chart_type, prompt)
                 message_data.update({"data": df, "show_chart": is_chart_request, "chart_type": chart_type, "query": prompt})
             else:
                 st.info("결과가 없습니다.")
                 message_data["content"] = "No data returned."
        else:
            st.error(f"실행 실패: {result_data}")
            message_data["error"] = f"Execution Failed: {result_data}"
        
        st.session_state.messages.append(message_data)
        st.session_state.sql_input = generated_sql

st.markdown("---")

# 수동 SQL 실행
with st.expander("📝 수동 SQL 실행", expanded=False):
    sql_input = st.text_area("SQL Query", value=st.session_state.sql_input, height=150)
    if st.button("▶️ 실행"):
        with st.spinner("실행 중..."):
            success, result_data = execute_tool_mcp("query", {"sql": sql_input})
            if success:
                if result_data:
                     df = pd.DataFrame(result_data)
                     display_data(df)
                else:
                    st.info("결과 없음")
            else:
                st.error(result_data)
