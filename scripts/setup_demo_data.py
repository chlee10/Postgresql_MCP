
import psycopg
import os
import random
import string
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

CONN_INFO = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER}"
if DB_PASSWORD:
    CONN_INFO += f" password={DB_PASSWORD}"

def get_connection():
    return psycopg.connect(CONN_INFO)

# =============================================================================
# Helper Functions
# =============================================================================

SURNAMES = {
    '김': 'kim', '이': 'lee', '박': 'park', '최': 'choi', '정': 'jung',
    '강': 'kang', '조': 'cho', '윤': 'yoon', '장': 'jang', '임': 'lim',
    '한': 'han', '오': 'oh', '서': 'seo', '신': 'shin', '권': 'kwon',
    '황': 'hwang', '안': 'ahn', '송': 'song', '류': 'ryu', '전': 'jeon',
    '홍': 'hong', '고': 'ko', '문': 'moon', '양': 'yang', '손': 'son',
    '배': 'bae', '백': 'baek', '허': 'hur', '유': 'yoo',
    '남': 'nam', '심': 'shim', '노': 'noh', '하': 'ha', '곽': 'kwak',
    '성': 'sung', '차': 'cha', '주': 'joo', '우': 'woo', '구': 'goo'
}

def generate_email(name, emp_id):
    last_char = name[0]
    eng_last = SURNAMES.get(last_char, 'user')
    initials = ''.join(random.choices(string.ascii_lowercase, k=2))
    return f"{eng_last}.{initials}{emp_id}@company.com"

def get_gender(name):
    # Simple heuristic
    last_char = name[-1]
    # Male-leaning
    if last_char in ['수', '준', '욱', '호', '영', '훈', '우', '섭', '석', '철']:
        return '남'
    # Female-leaning
    if last_char in ['희', '진', '경', '정', '은', '미', '지', '연', '현', '영', '숙']:
        return '여'
    return random.choice(['남', '여'])

def calculate_salary(rank, years_service):
    base_map = {
        '사원': 3200000,
        '대리': 4200000,
        '과장': 5500000,
        '차장': 6800000,
        '부장': 8000000,
        '팀장': 8800000
    }
    base = base_map.get(rank, 3000000)
    service_bonus = years_service * 150000
    random_factor = random.uniform(0.95, 1.05)
    return int((base + service_bonus) * random_factor / 10000) * 10000

# =============================================================================
# Steps
# =============================================================================

def step01_init_table(cur):
    print("STEP 1: Init Table & Basic Data...")
    cur.execute("DROP TABLE IF EXISTS 인사관리 CASCADE;")
    cur.execute("""
        CREATE TABLE 인사관리 (
            id SERIAL PRIMARY KEY,
            이름 VARCHAR(50),
            부서 VARCHAR(50),
            직급 VARCHAR(50),
            입사일 DATE,
            급여 INTEGER,
            전화번호 VARCHAR(20),
            이메일 VARCHAR(100),
            성별 VARCHAR(10),
            생년월일 DATE
        );
    """)
    # Basic data
    cur.execute("""
        INSERT INTO 인사관리 (이름, 부서, 직급, 입사일) VALUES
        ('김철수', '영업부', '과장', '2020-01-15'), ('이영희', '인사부', '대리', '2021-03-20'),
        ('박민준', '개발부', '과장', '2019-06-10'), ('정수진', '마케팅부', '사원', '2022-01-08'),
        ('최동욱', '재무부', '대리', '2021-09-15'), ('황미경', '영업부', '사원', '2022-05-01'),
        ('강준호', '개발부', '대리', '2020-11-03'), ('오수정', '인사부', '사원', '2023-02-14'),
        ('배준영', '영업부', '대리', '2021-07-22'), ('신현욱', '개발부', '사원', '2023-01-10'),
        ('윤지은', '마케팅부', '과장', '2020-04-18'), ('장서현', '재무부', '사원', '2022-08-25'),
        ('임태희', '영업부', '사원', '2023-03-05'), ('조민호', '개발부', '과장', '2019-02-14'),
        ('홍은정', '인사부', '대리', '2021-06-10'), ('성지훈', '마케팅부', '사원', '2022-10-03'),
        ('구은미', '영업부', '과장', '2018-11-20'), ('송준호', '개발부', '대리', '2021-05-15'),
        ('하지우', '재무부', '대리', '2020-09-08'), ('양민지', '마케팅부', '대리', '2021-12-01'),
        ('노영훈', '영업부', '대리', '2021-04-12'), ('류준영', '개발부', '사원', '2022-06-20'),
        ('김예진', '인사부', '사원', '2023-01-25'), ('이준호', '영업부', '사원', '2022-09-15'),
        ('박수현', '마케팅부', '사원', '2023-02-28'), ('정현우', '개발부', '대리', '2021-08-10'),
        ('최준영', '재무부', '과장', '2019-07-15'), ('황수진', '영업부', '사원', '2022-11-08'),
        ('강현정', '인사부', '대리', '2021-10-20'), ('오준호', '개발부', '사원', '2023-03-15'),
        ('배지영', '마케팅부', '사원', '2022-12-05'), ('신미경', '영업부', '과장', '2020-02-18'),
        ('윤준호', '개발부', '대리', '2021-01-12'), ('장현우', '재무부', '사원', '2023-01-30'),
        ('임지은', '마케팅부', '대리', '2021-11-22'), ('조수정', '영업부', '사원', '2022-07-18'),
        ('홍준영', '개발부', '과장', '2019-05-10'), ('성은미', '인사부', '사원', '2023-02-10'),
        ('구지훈', '영업부', '대리', '2021-03-25'), ('송수현', '마케팅부', '과장', '2020-08-15'),
        ('하준호', '개발부', '사원', '2022-04-08'), ('양현정', '재무부', '대리', '2021-09-20'),
        ('노지은', '영업부', '대리', '2021-08-01');
    """)

def step02_rename_teams(cur):
    print("STEP 2: Rename Departments to Teams...")
    cur.execute("UPDATE 인사관리 SET 부서 = REPLACE(부서, '부', '팀') WHERE 부서 LIKE '%부'")

def step03_new_employees_2026(cur):
    print("STEP 3: Add new employees (2026)...")
    new_names = [
        ("김신입", "남"), ("이새롬", "여"), ("박출발", "남"), ("최시작", "남"), ("정미래", "여"),
        ("강희망", "여"), ("조목표", "남"), ("윤도전", "남"), ("장성장", "여"), ("임패기", "남")
    ]
    cur.execute("SELECT DISTINCT 부서 FROM 인사관리")
    depts = [r[0] for r in cur.fetchall()]
    
    for name, gender in new_names:
        dept = random.choice(depts)
        join_date = date(2026, 1, 1) + timedelta(days=random.randint(0, 40))
        cur.execute("""
            INSERT INTO 인사관리 (이름, 부서, 직급, 입사일, 성별)
            VALUES (%s, %s, '사원', %s, %s)
        """, (name, dept, join_date, gender))

def step04_promote_ranks(cur):
    print("STEP 4: Promote Leaders (Team Lead, Manager)...")
    cur.execute("SELECT DISTINCT 부서 FROM 인사관리")
    depts = [r[0] for r in cur.fetchall()]
    
    for dept in depts:
        cur.execute("SELECT id, 이름 FROM 인사관리 WHERE 부서 = %s ORDER BY 입사일 ASC", (dept,))
        emp_list = cur.fetchall()
        
        # Team Lead (1st)
        if len(emp_list) >= 1:
            cur.execute("UPDATE 인사관리 SET 직급 = '팀장' WHERE id = %s", (emp_list[0][0],))
        # Manager (2nd)
        if len(emp_list) >= 2:
            cur.execute("UPDATE 인사관리 SET 직급 = '부장' WHERE id = %s", (emp_list[1][0],))
        # Senior (3rd, 4th)
        if len(emp_list) >= 3:
            count = 2 if len(emp_list) >= 4 else 1
            for i in range(count):
                cur.execute("UPDATE 인사관리 SET 직급 = '차장' WHERE id = %s", (emp_list[2+i][0],))

def step05_fill_details(cur):
    print("STEP 5: Fill details (Salary, Email, Phone, Age, Gender)...")
    cur.execute("SELECT id, 이름, 직급, 입사일, 성별, 생년월일 FROM 인사관리")
    rows = cur.fetchall()
    
    for row in rows:
        emp_id, name, rank, join_date, gender, birth = row
        
        # 1. Gender
        if not gender:
            gender = get_gender(name)
            
        # 2. Birth Date logic (At least 24 when joining)
        if not birth:
            # Base age: 25 at join
             join_year = join_date.year
             birth_year = join_year - 25
             birth = date(birth_year, random.randint(1,12), random.randint(1,28))
        else:
            # Re-validate existing birth
            if (join_date - birth).days / 365 < 24:
                birth = date(join_date.year - 25, random.randint(1,12), random.randint(1,28))

        # 3. Phone
        phone = f"010-{random.randint(2000,9999)}-{random.randint(2000,9999)}"
        
        # 4. Email
        email = generate_email(name, emp_id)
        
        # 5. Salary
        years = (date.today() - join_date).days / 365.25
        salary = calculate_salary(rank, years)
        
        cur.execute("""
            UPDATE 인사관리
            SET 성별 = %s, 생년월일 = %s, 전화번호 = %s, 이메일 = %s, 급여 = %s
            WHERE id = %s
        """, (gender, birth, phone, email, salary, emp_id))

def main():
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                step01_init_table(cur)
                step02_rename_teams(cur)
                step03_new_employees_2026(cur)
                step04_promote_ranks(cur)
                step05_fill_details(cur)
                conn.commit()
                print("\n✅ All steps completed! Database is ready.")
                
                # Report
                cur.execute("SELECT COUNT(*) FROM 인사관리")
                count = cur.fetchone()[0]
                print(f"Total Employees: {count}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
