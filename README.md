# PostgreSQL MCP AI Explorer

PostgreSQL 데이터베이스를 MCP(Model Context Protocol)로 연결하여,  
AI와 대화하며 자연어로 데이터를 탐색하고 시각화하는 프로젝트입니다.

---

## 🚀 주요 기능

- **자연어 → SQL 변환**: 복잡한 SQL 문법 없이 질문으로 데이터 조회
- **MCP 프로토콜 지원**: Claude Desktop 및 기타 MCP 클라이언트와 연동 가능
- **Streamlit Web UI**: 채팅 인터페이스 및 데이터 시각화 (차트 자동 생성)
- **실전형 데모 데이터**: 한국형 인사관리 데이터셋 자동 생성 스크립트 포함

## 🛠️ 설치 및 설정

### 1. 필수 요구사항
- Python 3.10 이상
- PostgreSQL 데이터베이스 (Local 또는 Docker)
- Poetry (패키지 관리자)

### 2. 설치
```bash
# 의존성 설치
poetry install
```

### 3. 환경 설정 (.env)
`.env` 파일을 생성하고 데이터베이스 정보를 입력하세요.
```ini
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password
```

## 🏃‍♂️ 실행 방법

### 1. 데이터베이스 초기화 (데모 데이터 생성)
실습을 위한 가상의 '인사관리' 테이블을 생성하고 50여 명의 사실적인 사원 정보를 채웁니다.
```bash
poetry run python scripts/setup_demo_data.py
```

### 2. MCP 서버 실행 (Backend)
Streamlit 앱이 통신할 백엔드 서버를 실행합니다.
```bash
poetry run python -m postgresql_mcp.server
```

### 3. Streamlit 앱 실행 (Frontend)
새 터미널을 열고 실행하세요.
```bash
poetry run streamlit run streamlit_app.py
```

## 🧪 테스트 쿼리 예시

- "개발팀의 평균 연봉은 얼마인가요?"
- "근속 3년 이상인 직원 중 급여 상위 5명 보여줘"
- "부서별 남녀 성비를 파이차트로 그려줘"
- "올해 입사한 신입사원 명단"

## 📁 프로젝트 구조

```text
PostgreSQL_MCP/
├── postgresql_mcp/       # MCP 서버 소스코드
│   └── server.py         # 메인 서버 엔트리포인트 (Windows Async Fix 포함)
├── scripts/              # 데이터 관리 스크립트
│   └── setup_demo_data.py # 데모 데이터 생성 마스터 스크립트
├── streamlit_app.py      # 웹 UI 애플리케이션
├── config.py             # 설정 파일 (프롬프트, 모델 등)
└── pyproject.toml        # 의존성 관리
```
