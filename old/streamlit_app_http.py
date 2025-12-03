"""
SQLcl AI Explorer - HTTP Server 모드

HTTP 서버에 연결하여 빠른 쿼리 실행
서버가 이미 DB에 로그인되어 있으므로 첫 쿼리도 빠름 (~0.2초)

사용법:
    1. 서버 실행: python -m sqlcl_mcp.http_server
    2. 앱 실행: streamlit run streamlit_app_http.py
"""

import streamlit as st
import os
import requests
import pandas as pd
import re
import time
import logging
from io import StringIO

from config import (
    SERVER_URL, AI_MODELS, MODEL_DISPLAY, PAGE_CONFIG_HTTP,
    DB_SCHEMA_INFO, SQL_GENERATION_RULES,
    CHART_KEYWORDS, LINE_CHART_KEYWORDS, PIE_CHART_KEYWORDS, AREA_CHART_KEYWORDS,
    APP_VERSION
)

# =============================================================================
# Optional Imports
# =============================================================================
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
st.set_page_config(**PAGE_CONFIG_HTTP)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sqlcl-http-client")


# =============================================================================
# Session State
# =============================================================================
def init_session_state():
    """세션 상태 초기화"""
    defaults = {
        'history': [],
        'sql_input': "",
        'messages': [],
        'selected_model': "claude-sonnet-4-5-20250929",
        'query_times': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# =============================================================================
# HTTP Client
# =============================================================================
def check_server_status() -> dict:
    """서버 상태 확인"""
    try:
        resp = requests.get(f"{SERVER_URL}/status", timeout=2)
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"connected": False, "error": "Server not running"}
    except Exception as e:
        return {"connected": False, "error": str(e)}


def execute_sql_http(sql: str, timeout: float = 60) -> tuple[bool, str, float]:
    """HTTP 서버를 통해 SQL 실행"""
    try:
        start = time.time()
        resp = requests.post(
            f"{SERVER_URL}/execute",
            json={"sql": sql, "timeout": timeout},
            timeout=timeout + 5
        )
        elapsed = time.time() - start
        data = resp.json()
        server_elapsed = data.get("elapsed", elapsed)
        
        if data.get("success"):
            return True, data.get("data", ""), server_elapsed
        return False, data.get("error", "Unknown error"), server_elapsed
        
    except requests.exceptions.ConnectionError:
        return False, "❌ 서버에 연결할 수 없습니다. 먼저 서버를 실행하세요:\npython -m sqlcl_mcp.http_server", 0
    except requests.exceptions.Timeout:
        return False, "Query timeout", timeout
    except Exception as e:
        return False, str(e), 0


# =============================================================================
# Display Functions
# =============================================================================
def display_data(df: pd.DataFrame, show_chart: bool = False, chart_type: str = "bar"):
    """데이터 표시 (테이블 + 차트)"""
    df_clean = df.dropna(axis=1, how='all')
    
    for col in df_clean.select_dtypes(include=['float64', 'float32']).columns:
        df_clean[col] = df_clean[col].round(1)
    
    df_display = df_clean.reset_index(drop=True)
    df_display.index = df_display.index + 1
    df_display.index.name = "No"
    
    if len(df_clean) == 1 and not show_chart:
        st.markdown("### 📋 상세 정보")
        for col in df_clean.columns:
            val = df_clean.iloc[0][col]
            if pd.notna(val) and str(val).strip():
                formatted = f"{val:.1f}" if isinstance(val, float) else val
                st.markdown(f"- **{col}**: {formatted}")
        return
    
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


def display_chart(df: pd.DataFrame, chart_type: str = "bar"):
    """차트 표시"""
    df_chart = df.copy()
    
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
    
    chart_funcs = {"bar": st.bar_chart, "line": st.line_chart, "area": st.area_chart}
    
    if chart_type in chart_funcs:
        chart_funcs[chart_type](chart_df, use_container_width=True)
    elif chart_type == "pie":
        try:
            import plotly.express as px
            fig = px.pie(df_chart, names=text_cols[0] if text_cols else None, values=value_col)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(chart_df, use_container_width=True)
    else:
        st.bar_chart(chart_df, use_container_width=True)


# =============================================================================
# Chart Detection
# =============================================================================
def detect_chart_request(query: str) -> tuple[bool, str]:
    """차트 요청 감지"""
    query_lower = query.lower()
    
    if '원그래프' in query_lower:
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
# SQL Helpers
# =============================================================================
@st.cache_data(ttl=3600)
def get_table_list() -> list:
    """테이블 목록 조회"""
    success, output, _ = execute_sql_http("SELECT table_name FROM user_tables ORDER BY table_name")
    if success:
        try:
            df = pd.read_csv(StringIO(output))
            col = 'TABLE_NAME' if 'TABLE_NAME' in df.columns else 'table_name'
            return df[col].tolist() if col in df.columns else []
        except Exception:
            pass
    return []


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


# =============================================================================
# SQL Generation (AI)
# =============================================================================
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
            content = msg.get("content", "")
            if not content:
                content = f"SQL: {msg['sql']}" if "sql" in msg else "OK"
            messages.append({"role": msg["role"], "content": content})
    messages.append({"role": "user", "content": nl_query})
    return messages


def _generate_sql_claude(nl_query: str, system_prompt: str, model_name: str, chat_history=None) -> str:
    """Claude로 SQL 생성"""
    if not HAS_ANTHROPIC:
        return "-- anthropic not installed"
    
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "-- ANTHROPIC_API_KEY not set"
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_name, max_tokens=2048, system=system_prompt,
            messages=_build_messages(chat_history, nl_query)
        )
        return clean_sql_response(response.content[0].text.strip())
    except Exception as e:
        return f"-- Error: {str(e)}"


def _generate_sql_openai(nl_query: str, system_prompt: str, model_name: str, chat_history=None) -> str:
    """OpenAI로 SQL 생성"""
    if not HAS_OPENAI:
        return "-- openai not installed"
    
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "-- OPENAI_API_KEY not set"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        messages = [{"role": "system", "content": system_prompt}] + _build_messages(chat_history, nl_query)
        response = client.chat.completions.create(model=model_name, messages=messages)
        return clean_sql_response(response.choices[0].message.content.strip())
    except Exception as e:
        return f"-- Error: {str(e)}"


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    # 헤더
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 3rem;">⚡</span>
        <h2 style="margin: 0.5rem 0 0 0;">SQLcl AI Explorer</h2>
        <p style="color: #888; font-size: 0.8rem;">HTTP Server Mode</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 서버 상태
    status = check_server_status()
    
    if status.get("connected"):
        uptime = status.get("uptime_seconds", 0)
        uptime_str = f"{int(uptime // 60)}분 {int(uptime % 60)}초" if uptime >= 60 else f"{uptime:.0f}초"
        query_count = status.get("query_count", 0)
        db_info = status.get("database", "Unknown")
        avg_time = sum(st.session_state.query_times[-10:]) / len(st.session_state.query_times[-10:]) if st.session_state.query_times else 0
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a472a, #2d5a3d); padding: 1rem; border-radius: 10px;">
            <span style="font-size: 1.2rem;">⚡</span>
            <span style="color: #90EE90; font-weight: 600;">서버 연결됨</span>
            <div style="color: #ccc; font-size: 0.75rem; margin-top: 0.5rem; line-height: 1.6;">
                📍 {db_info}<br>
                ⏱️ 서버 가동: {uptime_str}<br>
                📊 총 쿼리: {query_count}회<br>
                🚀 평균 응답: {avg_time:.3f}초
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4a1a1a, #5a2d2d); padding: 1rem; border-radius: 10px;">
            <span style="font-size: 1.2rem;">🔴</span>
            <span style="color: #FF6B6B; font-weight: 600;">서버 연결 안됨</span>
            <div style="color: #ccc; font-size: 0.75rem; margin-top: 0.5rem;">
                {status.get("error", "Unknown")}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.code("python -m sqlcl_mcp.http_server", language="bash")
    
    if st.button("🔄 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    # AI 모델 선택
    st.markdown("##### 🤖 AI 모델")
    st.selectbox(
        "Model", AI_MODELS, index=0, key="selected_model",
        label_visibility="collapsed",
        format_func=lambda x: f"{MODEL_DISPLAY.get(x, ('', x, ''))[0]} {MODEL_DISPLAY.get(x, ('', x, ''))[1]}"
    )
    
    st.markdown("---")
    
    with st.expander("📈 성능", expanded=False):
        st.markdown("""
        **HTTP 서버 모드 장점:**
        - 서버 시작시 DB 로그인 완료
        - 첫 쿼리도 빠름 (~0.2초)
        - 서버 독립 실행 가능
        """)
    
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.7rem;">
        {APP_VERSION} • HTTP Server Mode
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Main Interface
# =============================================================================
st.markdown("## ⚡ SQLcl AI Explorer")
st.caption("HTTP 서버 모드 - 서버가 먼저 DB에 연결되어 모든 쿼리가 빠릅니다")

# 채팅 히스토리 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            if "sql" in msg:
                with st.expander("🔍 SQL", expanded=False):
                    st.code(msg["sql"], language="sql")
            if "elapsed" in msg:
                st.caption(f"⚡ {msg['elapsed']:.3f}초")
            if "data" in msg:
                display_data(msg["data"], msg.get("show_chart", False), msg.get("chart_type", "bar"))
            elif "error" in msg:
                st.error(msg["error"])
            elif "content" in msg:
                st.write(msg["content"])
        else:
            st.write(msg["content"])

# 채팅 입력
if prompt := st.chat_input("질문하세요 (예: 부서별 인원수)"):
    if not check_server_status().get("connected"):
        st.error("❌ 서버가 실행 중이지 않습니다. 먼저 서버를 시작하세요:\n\n`python -m sqlcl_mcp.http_server`")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    is_chart_request, chart_type = detect_chart_request(prompt)

    with st.chat_message("assistant"):
        with st.spinner("SQL 생성 중..."):
            tables = get_table_list()
            generated_sql = generate_sql_from_nl(prompt, tables, st.session_state.selected_model, st.session_state.messages[:-1])
            
            with st.expander("🔍 SQL", expanded=False):
                st.code(generated_sql, language="sql")
            
            if not is_valid_sql(generated_sql):
                st.warning("유효한 SQL을 생성하지 못했습니다.")
                st.session_state.messages.append({"role": "assistant", "content": "SQL 생성 실패", "sql": generated_sql})
                st.stop()
            
            success, output, elapsed = execute_sql_http(generated_sql)
            st.session_state.query_times.append(elapsed)
            st.caption(f"⚡ {elapsed:.3f}초")
            
            message_data = {"role": "assistant", "sql": generated_sql, "elapsed": elapsed}
            
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
                    message_data["error"] = str(e)
            else:
                st.error("실행 실패")
                st.code(output)
                message_data["error"] = output
            
            st.session_state.messages.append(message_data)

st.markdown("---")

# 수동 SQL 실행
with st.expander("📝 수동 SQL 실행", expanded=False):
    sql_input = st.text_area("SQL", value=st.session_state.sql_input, height=150)
    if st.button("▶️ 실행"):
        if not check_server_status().get("connected"):
            st.error("서버가 실행 중이지 않습니다.")
        else:
            success, output, elapsed = execute_sql_http(sql_input)
            st.caption(f"⚡ {elapsed:.3f}초")
            
            if success:
                try:
                    df = pd.read_csv(StringIO(output))
                    display_data(df)
                except Exception:
                    st.code(output)
            else:
                st.error(output)
