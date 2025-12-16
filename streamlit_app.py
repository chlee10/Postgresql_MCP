"""
SQLcl AI Explorer - 자연어로 Oracle DB 탐색 (MCP SSE 기반)

사용법:
    1. MCP 서버 시작: poetry run python -m sqlcl_mcp.server --sse
    2. Streamlit 앱: streamlit run streamlit_app.py
"""

import streamlit as st
import os
import asyncio
import pandas as pd
import re
import logging
import httpx
from io import StringIO

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
        'selected_model': DEFAULT_MODEL
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# =============================================================================
# SQL Column Alias Extraction
# =============================================================================
def extract_column_aliases(sql: str) -> list:
    """
    SQL SELECT 문에서 컬럼 별칭(alias) 추출
    예: SELECT COUNT(*) AS CNT, NAME AS 이름 FROM ... -> ['CNT', '이름']
    """
    sql_upper = sql.upper()
    
    # SELECT ~ FROM 사이의 컬럼 부분 추출
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_upper, re.DOTALL | re.IGNORECASE)
    if not select_match:
        return []
    
    # 원본 SQL에서도 같은 위치 추출 (대소문자 유지를 위해)
    orig_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.DOTALL | re.IGNORECASE)
    if not orig_match:
        return []
    
    columns_part = orig_match.group(1)
    
    # 괄호 안의 쉼표는 무시하고 컬럼 분리
    aliases = []
    depth = 0
    current = ""
    for char in columns_part:
        if char == '(':
            depth += 1
            current += char
        elif char == ')':
            depth -= 1
            current += char
        elif char == ',' and depth == 0:
            aliases.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        aliases.append(current.strip())
    
    # 각 컬럼에서 AS 뒤의 별칭 추출
    result = []
    for col_expr in aliases:
        # AS 키워드로 별칭 찾기
        as_match = re.search(r'\s+AS\s+(["\']?)(\w+)\1\s*$', col_expr, re.IGNORECASE)
        if as_match:
            result.append(as_match.group(2))
        else:
            # AS 없이 공백으로 구분된 별칭 (예: COUNT(*) CNT)
            parts = col_expr.strip().split()
            if len(parts) >= 2:
                last_part = parts[-1].strip('"\'')
                # 마지막이 순수 식별자인 경우만 별칭으로 인식
                if re.match(r'^[\w가-힣]+$', last_part):
                    result.append(last_part)
                else:
                    # 함수나 컬럼명 자체 사용
                    result.append(_extract_simple_name(col_expr))
            else:
                result.append(_extract_simple_name(col_expr))
    
    return result


def _extract_simple_name(expr: str) -> str:
    """표현식에서 간단한 이름 추출"""
    expr = expr.strip()
    # 테이블.컬럼 형식에서 컬럼만
    if '.' in expr:
        expr = expr.split('.')[-1]
    # 함수 호출에서 함수명
    if '(' in expr:
        func_match = re.match(r'(\w+)\s*\(', expr)
        if func_match:
            return func_match.group(1)
    return expr.strip('"\'')


def parse_csv_with_headers(csv_output: str, sql: str) -> pd.DataFrame:
    """
    CSV 출력을 DataFrame으로 변환 (SQL에서 헤더 추출)
    """
    if not csv_output.strip():
        return pd.DataFrame()
    
    # SQL에서 컬럼 별칭 추출
    headers = extract_column_aliases(sql)
    
    lines = csv_output.strip().split('\n')
    if not lines:
        return pd.DataFrame()
    
    # 첫 번째 줄로 컬럼 개수 확인
    first_line = lines[0]
    # CSV 파싱으로 컬럼 개수 확인
    try:
        test_df = pd.read_csv(StringIO(first_line), header=None)
        num_cols = len(test_df.columns)
    except Exception:
        num_cols = len(headers) if headers else 1
    
    # 헤더 개수와 컬럼 개수 맞추기
    if len(headers) != num_cols:
        headers = [f"COL{i+1}" for i in range(num_cols)]
    
    try:
        df = pd.read_csv(StringIO(csv_output), header=None, names=headers, quotechar='"', on_bad_lines='warn')
        
        # 첫 번째 행이 헤더와 동일한 경우 제거 (SQLcl이 헤더를 데이터로 포함하는 경우)
        if len(df) > 0:
            first_row = df.iloc[0].astype(str).str.strip().str.upper().tolist()
            header_upper = [str(h).strip().upper() for h in headers]
            if first_row == header_upper:
                df = df.iloc[1:].reset_index(drop=True)
                logger.info("Removed duplicate header row from CSV data")
        
        return df
    except Exception as e:
        logger.warning(f"CSV parsing error: {e}")
        return pd.DataFrame()


# =============================================================================
# Display Functions
# =============================================================================
def display_data(df: pd.DataFrame, show_chart: bool = False, chart_type: str = "bar", query: str = ""):
    """데이터 표시 (테이블 + 차트)"""
    df_clean = df.dropna(axis=1, how='all')
    
    # 숫자 포맷팅 - float를 소수점 1자리 문자열로 변환
    for col in df_clean.select_dtypes(include=['float64', 'float32']).columns:
        df_clean[col] = df_clean[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
    
    # 인덱스 1부터 시작
    df_display = df_clean.reset_index(drop=True)
    df_display.index = df_display.index + 1
    df_display.index.name = "No"
    
    # 단일 행: 상세 정보 표시
    if len(df_clean) == 1 and not show_chart:
        st.markdown("#### 📋 상세 정보")
        detail_html = "<div style='line-height: 1.4; margin: 0;'>"
        for col in df_clean.columns:
            val = df_clean.iloc[0][col]
            if pd.notna(val) and str(val).strip():
                formatted = f"{val:.1f}" if isinstance(val, float) else val
                detail_html += f"<div style='margin: 2px 0;'>• <b>{col}</b>: {formatted}</div>"
        detail_html += "</div>"
        st.markdown(detail_html, unsafe_allow_html=True)
        return
    
    # 차트 표시
    if show_chart and len(df_clean) > 0:
        display_chart(df_clean, chart_type, query)
    else:
        # 표만 나올 때 제목 추가
        table_title = extract_chart_title(query) if query else "조회 결과"
        st.markdown(f"#### 📋 {table_title}")
    
    # 테이블 스타일
    st.markdown("""
    <style>
        div[data-testid="stTable"] table { width: 100%; }
        div[data-testid="stTable"] th { text-align: center !important; background-color: #f0f2f6; padding: 4px 10px !important; }
        div[data-testid="stTable"] td { text-align: center !important; padding: 2px 8px !important; }
        div[data-testid="stTable"] tr { line-height: 1.2 !important; }
    </style>
    """, unsafe_allow_html=True)
    st.table(df_display)


def display_chart(df: pd.DataFrame, chart_type: str = "bar", query: str = ""):
    """차트 표시"""
    import uuid
    
    df_chart = df.copy()
    
    # 첫 번째 컬럼은 라벨로 사용 (숫자여도 문자열로 유지)
    first_col = df_chart.columns[0]
    df_chart[first_col] = df_chart[first_col].astype(str)
    
    # 나머지 컬럼만 숫자 변환
    for col in df_chart.columns[1:]:
        df_chart[col] = pd.to_numeric(df_chart[col], errors='ignore')
    
    text_cols = df_chart.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df_chart.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        st.warning("차트를 그릴 수치 데이터가 없습니다.")
        return
    
    label_col = text_cols[0] if text_cols else "Index"
    value_col = numeric_cols[0]
    
    # 쿼리에서 차트 제목 추출
    chart_title = extract_chart_title(query) if query else f"{value_col} by {label_col}"
    st.markdown(f"#### 📊 {chart_title}")
    
    # Plotly로 모든 차트 그리기 (데이터 순서 유지)
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
            # X축 카테고리 순서 유지
            fig.update_xaxes(categoryorder='array', categoryarray=df_chart[label_col].tolist())
        
        # 고유한 key로 차트 ID 충돌 방지
        chart_key = f"chart_{uuid.uuid4().hex[:8]}"
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
    except ImportError:
        # Plotly 없으면 기본 Streamlit 차트 (순서 유지 안됨)
        chart_df = df_chart.set_index(text_cols[0])[[value_col]] if text_cols else df_chart[[value_col]]
        chart_funcs = {"bar": st.bar_chart, "line": st.line_chart, "area": st.area_chart}
        chart_funcs.get(chart_type, st.bar_chart)(chart_df, use_container_width=True)


def extract_chart_title(query: str) -> str:
    """쿼리에서 차트 제목 추출"""
    import re
    
    title = query.strip()
    
    # 0. 쉼표/마침표 이후 부가 조건 제거 (정렬, 필터 조건 등)
    # 예: "부서별 인원수를 보여줘, 근속 년수가 많은 순으로" → "부서별 인원수를 보여줘"
    title = re.split(r'[,.]', title)[0].strip()
    
    # 1. 차트/그래프 관련 전체 구문 제거
    chart_patterns = [
        r'을\s*(원그래프|파이차트|막대그래프|바차트|라인차트|라인그래프|영역차트|막대|라인)(로|으로)?\s*(그려줘|보여줘|표시해줘)?',
        r'를\s*(원그래프|파이차트|막대그래프|바차트|라인차트|라인그래프|영역차트|막대|라인)(로|으로)?\s*(그려줘|보여줘|표시해줘)?',
        r'(원그래프|파이차트|막대그래프|바차트|라인차트|라인그래프|영역차트)(로|으로)?\s*(그려줘|보여줘|표시해줘)?',
        r'(차트|그래프)(로|으로)?\s*(그려줘|보여줘)?',
    ]
    for pattern in chart_patterns:
        title = re.sub(pattern, '', title, flags=re.IGNORECASE)
    
    # 2. 끝에서부터 불용어 반복 제거 (긴 것부터 먼저!)
    stopwords = [
        '함께 보여줘', '같이 보여줘', '함께 알려줘', '같이 알려줘',
        '보여줘', '알려줘', '조회해줘', '표시해줘', '만들어줘', '그려줘',
        '해줘', '함께', '같이', '좀',
        '를', '을', '로', '으로', '줘', '보여'
    ]
    
    changed = True
    while changed:
        changed = False
        title = title.strip()
        for word in stopwords:
            if title.endswith(word):
                title = title[:-len(word)]
                changed = True
                break
    
    # 공백 정리
    title = ' '.join(title.split()).strip()
    
    return title if title else query


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
def execute_sql_via_mcp(sql_query: str) -> tuple[bool, str]:
    """SSE MCP 서버를 통한 SQL 실행 (상주 서버, DB 연결 유지)"""
    if not HAS_MCP:
        return False, "mcp package not installed"

    async def _run():
        # SseServerTransport는 /sse 경로에서 SSE 연결을 시작
        sse_url = f"{SERVER_URL}/sse"
        async with sse_client(sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool("execute_sql", arguments={"sql": sql_query})

    try:
        result = asyncio.run(_run())
        if not result.content:
            return False, "No content returned"
        text = result.content[0].text
        return (False, text) if text.startswith("ERROR:") or "ORA-" in text else (True, text)
    except httpx.ConnectError:
        return False, "MCP 서버에 연결할 수 없습니다. 서버를 먼저 시작하세요:\npoetry run python -m sqlcl_mcp.server"
    except Exception as e:
        return False, f"MCP Error: {str(e)}"


def execute_sql(sql_query: str) -> tuple[bool, str]:
    """MCP 서버를 통한 SQL 실행"""
    return execute_sql_via_mcp(sql_query)


@st.cache_data(ttl=3600)
def get_table_list() -> list:
    """테이블 목록 조회 (캐시됨)"""
    sql = "SELECT table_name AS TABLE_NAME FROM user_tables ORDER BY table_name"
    success, output = execute_sql(sql)
    if success:
        try:
            df = parse_csv_with_headers(output, sql)
            col = 'TABLE_NAME' if 'TABLE_NAME' in df.columns else df.columns[0] if len(df.columns) > 0 else None
            return df[col].tolist() if col else []
        except Exception:
            pass
    return []


# =============================================================================
# SQL Generation (AI)
# =============================================================================
def clean_sql_response(content: str) -> str:
    """AI 응답에서 SQL 추출"""
    # 1. Markdown Code Block 추출
    if "```" in content:
        match = re.search(r"```(?:sql)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
            
    # 2. SQL 키워드로 시작하는 부분 찾기
    valid_starts = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]
    upper = content.upper()
    
    # 이미 SQL로 시작하면 그대로 진행
    if any(upper.startswith(k) for k in valid_starts):
        pass
    else:
        # 중간에 SQL이 있는지 찾기
        pattern = r"(" + "|".join(valid_starts) + r")\s+.*"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(0)
    
    # 3. 불필요한 텍스트 제거
    content = re.sub(r';\s*\w+\s*$', ';', content)
    content = re.sub(r'\s+(OK|Done|Success|완료)\.?\s*$', '', content, flags=re.IGNORECASE)
    
    return content.strip()


def is_valid_sql(content: str) -> bool:
    """SQL 유효성 검사"""
    if not content:
        return False
    
    # 마크다운 볼드체 등 제거 및 공백 제거
    clean_content = content.replace('**', '').replace('*', '').strip()
    
    # 물음표로 끝나면 SQL이 아닐 확률이 높음 (대화형 질문)
    if clean_content.endswith('?'):
        return False
        
    upper = content.strip().upper()
    valid_starts = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]
    
    if not any(upper.startswith(k) for k in valid_starts):
        return False
        
    # WITH로 시작하는 경우, CTE인지 일반 문장인지 구별
    if upper.startswith("WITH"):
        # CTE는 보통 "WITH 이름 AS" 형태임
        # "WITH" 뒤에 공백이 있고, 그 뒤에 식별자가 오고, 그 뒤에 "AS"가 와야 함
        # 간단하게 " AS " 또는 " AS(" 가 초반에 나오는지 확인
        # "with a specific query?" 에는 " AS "가 없음
        # 줄바꿈 후 AS가 올 수도 있음
        check_range = upper[:100] # 처음 100자만 확인
        if " AS " not in check_range and " AS(" not in check_range and "\nAS" not in check_range:
            return False
            
    return True


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
        <span style="font-size: 2.5rem;">🔶</span>
        <h2 style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">Oracle MCP Server</h2>
        <p style="color: #888; font-size: 0.75rem; margin: 0.3rem 0 0 0;">Implementation</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # AI 모델 선택
    st.markdown("##### 🤖 AI Model")
    st.selectbox(
        "Model", AI_MODELS, key="selected_model",
        label_visibility="collapsed",
        format_func=lambda x: f"{MODEL_DISPLAY.get(x, ('', x, ''))[0]} {MODEL_DISPLAY.get(x, ('', x, ''))[1]}"
    )
    
    st.markdown("---")
    
    # 푸터
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.7rem;">
        MCP + SQLcl<br>
        <span style="font-size: 0.65rem;">v{APP_VERSION}</span>
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
        # Placeholder로 이전 결과 잔상 방지
        result_placeholder = st.empty()
        
        with result_placeholder.container():
            with st.spinner("SQL 생성 및 실행 중..."):
                tables = get_table_list()
                history = st.session_state.messages[:-1]
                generated_sql = generate_sql_from_nl(prompt, tables, st.session_state.selected_model, history)
        
        # 결과 표시
        result_placeholder.empty()
        
        with st.expander("🔍 View SQL", expanded=False):
            st.code(generated_sql, language="sql")
        
        if not is_valid_sql(generated_sql):
            st.warning("유효한 SQL을 생성하지 못했습니다.")
            st.code(generated_sql)
            st.session_state.messages.append({"role": "assistant", "content": "SQL 생성 실패", "sql": generated_sql})
            st.stop()
        
        with st.spinner("쿼리 실행 중..."):
            success, output = execute_sql(generated_sql)
        
        message_data = {"role": "assistant", "sql": generated_sql}
        
        if success:
            try:
                if output.strip():
                    df = parse_csv_with_headers(output, generated_sql)
                    if not df.empty:
                        display_data(df, is_chart_request, chart_type, prompt)
                        message_data.update({"data": df, "show_chart": is_chart_request, "chart_type": chart_type, "query": prompt})
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
                    df = parse_csv_with_headers(output, sql_input)
                    display_data(df) if not df.empty else st.warning("데이터 없음")
                except Exception:
                    st.code(output)
            else:
                st.error(output)
