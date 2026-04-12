import os
import random
import json
import re
from datetime import datetime, timedelta
import smtplib
from email.message import EmailMessage
import joblib
import mysql.connector
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

APP_SECRET = os.environ.get("APP_SECRET", "svp_secret_dev")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "Root@123")
DB_NAME = os.environ.get("DB_NAME", "college_chatbot")
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
SMTP_FROM = os.environ.get("SMTP_FROM", SMTP_USER)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "intent_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
MODEL_CACHE = {"model": None, "vectorizer": None, "encoder": None, "mtime": None}

CONFIDENCE_THRESHOLD = 0.4
SYLLABUS_STORE_PATH = os.path.join("data", "syllabus_store.json")
SYLLABUS_CACHE = None
SYLLABUS_MTIME = None

app = Flask(__name__)
app.secret_key = APP_SECRET


def get_db_connection(use_db=True):
    config = {
        "host": DB_HOST,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "autocommit": True
    }
    if use_db:
        config["database"] = DB_NAME
    return mysql.connector.connect(**config)


def init_db():
    schema_path = "schema.sql"
    if not os.path.exists(schema_path):
        return
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        conn = get_db_connection(use_db=False)
        cursor = conn.cursor()
        for stmt in schema_sql.split(";"):
            s = stmt.strip()
            if s:
                cursor.execute(s)
        cursor.close()
        conn.close()
    except Exception:
        pass


def ensure_admin_user():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM admin_users")
        count = cursor.fetchone()[0]
        if count == 0:
            default_user = "admin"
            default_pass = "admin123"
            password_hash = generate_password_hash(default_pass)
            cursor.execute(
                "INSERT INTO admin_users (username, password_hash) VALUES (%s, %s)",
                (default_user, password_hash)
            )
        cursor.close()
        conn.close()
    except Exception:
        pass


def load_model_bundle():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(ENCODER_PATH)):
        return None, None, None
    try:
        mtimes = (
            os.path.getmtime(MODEL_PATH),
            os.path.getmtime(VECTORIZER_PATH),
            os.path.getmtime(ENCODER_PATH),
        )
    except OSError:
        return None, None, None
    if MODEL_CACHE["mtime"] == mtimes:
        return MODEL_CACHE["model"], MODEL_CACHE["vectorizer"], MODEL_CACHE["encoder"]
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    MODEL_CACHE["model"] = model
    MODEL_CACHE["vectorizer"] = vectorizer
    MODEL_CACHE["encoder"] = encoder
    MODEL_CACHE["mtime"] = mtimes
    return model, vectorizer, encoder


def predict_intent(message: str):
    model, vectorizer, encoder = load_model_bundle()
    if model is None:
        return None, 0.0
    X = vectorizer.transform([message])
    probs = model.predict_proba(X)[0]
    best_idx = probs.argmax()
    confidence = float(probs[best_idx])
    intent = encoder.inverse_transform([best_idx])[0]
    return intent, confidence


def fetch_answer(intent: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT answer FROM qa_pairs WHERE intent=%s", (intent,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        if rows:
            return random.choice(rows)[0]
    except Exception:
        return None
    return None


def log_chat(user_message, bot_response, intent=None, confidence=None):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_logs (user_message, bot_response, intent, confidence) VALUES (%s,%s,%s,%s)",
            (user_message, bot_response, intent, confidence)
        )
        cursor.close()
        conn.close()
    except Exception:
        pass


def fetch_student_by_roll(roll):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, roll, email, password_hash, is_verified, otp_code, otp_expires_at, verified_at "
            "FROM students WHERE roll=%s",
            (roll,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return row
    except Exception:
        return None


def fetch_student_by_email(email):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, roll, email, password_hash, is_verified, otp_code, otp_expires_at, verified_at "
            "FROM students WHERE email=%s",
            (email,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return row
    except Exception:
        return None


def create_student(name, roll, email, password_hash, otp_code, otp_expires_at):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO students (name, roll, username, email, password_hash, otp_code, otp_expires_at, is_verified) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,0)",
            (name, roll, roll, email, password_hash, otp_code, otp_expires_at)
        )
        cursor.close()
        conn.close()
        return True, None
    except Exception as e:
        return False, str(e)


def update_student_pending_by_roll(name, roll, email, password_hash, otp_code, otp_expires_at):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE students SET name=%s, username=%s, email=%s, password_hash=%s, otp_code=%s, otp_expires_at=%s, is_verified=0 "
            "WHERE roll=%s",
            (name, roll, email, password_hash, otp_code, otp_expires_at, roll)
        )
        cursor.close()
        conn.close()
        return True, None
    except Exception as e:
        return False, str(e)


def mark_student_verified(roll):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE students SET is_verified=1, verified_at=%s, otp_code=NULL, otp_expires_at=NULL WHERE roll=%s",
            (datetime.now(), roll)
        )
        cursor.close()
        conn.close()
        return True
    except Exception:
        return False


def send_otp_email(to_email, name, otp_code):
    if not (SMTP_USER and SMTP_PASS and SMTP_FROM):
        return False, "SMTP is not configured. Please set SMTP_USER, SMTP_PASS, SMTP_FROM."
    try:
        msg = EmailMessage()
        msg["Subject"] = "Your OTP for AI/ML Chatbot Registration"
        msg["From"] = SMTP_FROM
        msg["To"] = to_email
        msg.set_content(
            f"Hello {name},\n\n"
            f"Your OTP for registration is: {otp_code}\n\n"
            "This OTP is valid for 10 minutes.\n"
            "If you did not request this, please ignore this email.\n\n"
            "Regards,\nSVP AI/ML Chatbot"
        )
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        return True, None
    except Exception as e:
        return False, str(e)


def format_timestamp(ts):
    if not ts:
        ts = datetime.now()
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except Exception:
            ts = datetime.now()
    return ts.strftime("%d %b %Y, %I:%M %p")


def fetch_intent_distribution(limit=5):
    demo = [
        {"label": "Admissions", "pct": 78},
        {"label": "Courses", "pct": 64},
        {"label": "Fees", "pct": 51},
        {"label": "Placements", "pct": 46},
        {"label": "Facilities", "pct": 39},
    ]
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT intent, COUNT(*) AS cnt FROM chat_logs "
            f"WHERE intent IS NOT NULL GROUP BY intent ORDER BY cnt DESC LIMIT {int(limit)}"
        )
        rows = cursor.fetchall()
        cursor.execute("SELECT MAX(created_at) FROM chat_logs")
        last_updated = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        if not rows:
            return demo, format_timestamp(last_updated)
        total = sum(r[1] for r in rows) or 0
        if total == 0:
            return demo, format_timestamp(last_updated)
        data = [{"label": r[0], "pct": round(r[1] * 100 / total)} for r in rows]
        return data, format_timestamp(last_updated)
    except Exception:
        return demo, format_timestamp(None)


def fetch_intent_trend(days=7):
    demo = [
        {"label": "Mon", "value": 12, "pct": 55},
        {"label": "Tue", "value": 18, "pct": 75},
        {"label": "Wed", "value": 9, "pct": 42},
        {"label": "Thu", "value": 22, "pct": 90},
        {"label": "Fri", "value": 15, "pct": 65},
        {"label": "Sat", "value": 6, "pct": 28},
        {"label": "Sun", "value": 10, "pct": 48},
    ]
    try:
        days = int(days)
        if days < 2:
            days = 7
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days - 1)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DATE(created_at) AS d, COUNT(*) AS cnt "
            "FROM chat_logs WHERE intent IS NOT NULL AND created_at >= %s "
            "GROUP BY d ORDER BY d",
            (start_date,)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        count_map = {r[0]: r[1] for r in rows}
        series = []
        max_count = max(count_map.values()) if count_map else 0
        if max_count == 0:
            return demo
        for i in range(days):
            d = start_date + timedelta(days=i)
            cnt = int(count_map.get(d, 0))
            pct = round(cnt * 100 / max_count) if max_count else 0
            series.append({"label": d.strftime("%a"), "value": cnt, "pct": pct})
        return series
    except Exception:
        return demo


def load_syllabus_store():
    global SYLLABUS_CACHE, SYLLABUS_MTIME
    try:
        mtime = os.path.getmtime(SYLLABUS_STORE_PATH)
    except OSError:
        SYLLABUS_CACHE = None
        SYLLABUS_MTIME = None
        return None

    if SYLLABUS_CACHE is not None and SYLLABUS_MTIME == mtime:
        return SYLLABUS_CACHE

    try:
        with open(SYLLABUS_STORE_PATH, "r", encoding="utf-8") as f:
            SYLLABUS_CACHE = json.load(f)
        SYLLABUS_MTIME = mtime
        return SYLLABUS_CACHE
    except Exception:
        SYLLABUS_CACHE = None
        SYLLABUS_MTIME = None
        return None


def detect_curriculum(text: str):
    if re.search(r"\bc\s*[-_]?20\b", text):
        return "C20"
    if re.search(r"\bc\s*[-_]?25\b", text):
        return "C25"
    return None


def detect_branch(text: str):
    aliases = [
        ("CSE", ["cse", "computer science engineering", "computer science and engineering", "computer science"]),
        ("CS", ["cs"]),
        ("ECE", ["ece", "electronics and communication", "electronics & communication", "electronics communication engineering"]),
        ("EC", ["ec"]),
        ("DAE", ["dae", "diploma in automobile engineering", "automobile engineering diploma"]),
        ("AT", ["automobile", "automobile engineering"]),
        ("DME", ["dme", "diploma in mechanical engineering"]),
        ("ME", ["mechanical", "mechanical engineering"]),
    ]
    for key, keys in aliases:
        for alias in keys:
            if re.search(rf"\b{re.escape(alias)}\b", text):
                return key
    return None


def detect_sem_range(text: str):
    if re.search(r"\b1\s*(?:-|to|&)\s*4\b", text):
        return "1-4"
    if re.search(r"\b1\s*(?:-|to|&)\s*2\b", text) or re.search(r"\b(1st|first)\s*(?:and|&)\s*(2nd|second)\b", text):
        return "1-2"
    if re.search(r"\b3\s*(?:-|to|&)\s*4\b", text) or re.search(r"\b(3rd|third)\s*(?:and|&)\s*(4th|fourth)\b", text):
        return "3-4"
    if re.search(r"\b5\s*(?:-|to|&)\s*6\b", text) or re.search(r"\b(5th|fifth)\s*(?:and|&)\s*(6th|sixth)\b", text):
        return "5-6"

    if re.search(r"\b(1st|2nd|first|second|1|2)\b.*\b(sem|semester)\b", text):
        return "1-2"
    if re.search(r"\b(3rd|4th|third|fourth|3|4)\b.*\b(sem|semester)\b", text):
        return "3-4"
    if re.search(r"\b(5th|6th|fifth|sixth|5|6)\b.*\b(sem|semester)\b", text):
        return "5-6"

    return None


def detect_sem_number(text: str):
    match = re.search(r"\b(1st|2nd|3rd|4th|5th|6th)\b", text)
    if match:
        return {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6}[match.group(1)]
    match = re.search(r"\b([1-6])\s*(?:sem|semester)\b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(1|2|3|4|5|6)\b", text)
    if match and "sem" in text:
        return int(match.group(1))
    return None


def extract_sem_section(text: str, sem_number: int):
    if not sem_number:
        return text
    roman_map = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
    target_roman = roman_map.get(sem_number)
    if not target_roman:
        return text
    upper = text.upper()
    patterns = [
        rf"\b{target_roman}\s*SEM",
        rf"\b{sem_number}(?:ST|ND|RD|TH)?\s*SEM",
        rf"\bSEMESTER\s*{sem_number}\b",
    ]
    positions = []
    for idx, m in enumerate(re.finditer(r"\b(I{1,3}|IV|V|VI)\s*SEM\b|\b[1-6](?:ST|ND|RD|TH)?\s*SEM\b|\bSEMESTER\s*[1-6]\b", upper)):
        positions.append((m.start(), m.group(0)))

    start = None
    for pos, label in positions:
        if re.search(patterns[0], label) or re.search(patterns[1], label) or re.search(patterns[2], label):
            start = pos
            break
    if start is None:
        return text

    end = len(text)
    for pos, _ in positions:
        if pos > start:
            end = pos
            break
    return text[start:end]


def count_subject_lines(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    subject_lines = []
    pattern = re.compile(r"^[A-Z]{1,4}\s*[-]?\s*\d{2,4}[A-Z]?\s*[-:]\s+.+")
    for ln in lines:
        if pattern.match(ln):
            subject_lines.append(ln)
    seen = set()
    unique = []
    for ln in subject_lines:
        if ln not in seen:
            seen.add(ln)
            unique.append(ln)
    return unique


def extract_subject_names(text: str):
    subject_lines = count_subject_lines(text)
    names = []
    pattern = re.compile(r"^[A-Z]{1,4}\s*[-]?\s*\d{2,4}[A-Z]?\s*[-:]\s+(.+)")
    for ln in subject_lines:
        m = pattern.match(ln)
        if m:
            name = m.group(1).strip()
            names.append(name)
    return names


def extract_course_titles(text: str, sem_number=None):
    # Attempt to extract subject names from "Course Title" blocks.
    titles = []
    pattern = re.compile(
        r"Course Title\s*[:\-]?\s*(.+?)(?:Category|No\.\s*of\s*Credits|Type\s*of\s*Course|CIE\s*Marks|SEE\s*Marks|Total\s*Contact\s*Hours|Teaching\s*Scheme|$)",
        re.IGNORECASE | re.DOTALL
    )
    for m in pattern.finditer(text):
        raw = m.group(1)
        raw = re.sub(r"\s+", " ", raw).strip()
        raw = re.sub(r"\s*Course Group.*$", "", raw, flags=re.IGNORECASE).strip()
        if not raw:
            continue
        # Look back a bit for semester label to filter.
        window_start = max(0, m.start() - 300)
        window = text[window_start:m.start()]
        sem_ok = True
        if sem_number:
            roman_map = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
            roman = roman_map.get(sem_number)
            sem_ok = False
            if roman and re.search(rf"Semester\s*[:]*\s*{roman}\b", window, re.IGNORECASE):
                sem_ok = True
            if re.search(rf"\b{sem_number}(?:ST|ND|RD|TH)?\s*SEM", window, re.IGNORECASE):
                sem_ok = True
        if sem_ok:
            titles.append(raw)
    # De-dup while preserving order
    seen = set()
    unique = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def extract_relevant_snippets(text: str, message: str, limit=8):
    stop = {"the", "and", "are", "there", "how", "many", "what", "is", "for", "in", "of", "to", "a", "an"}
    words = [w for w in re.findall(r"[a-z0-9]+", message.lower()) if len(w) > 2 and w not in stop]
    if not words:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    hits = []
    for ln in lines:
        l = ln.lower()
        if any(w in l for w in words):
            hits.append(ln)
        if len(hits) >= limit:
            break
    return hits


def find_syllabus_response(message: str):
    text = message.lower()
    keyword_hits = ["syllabus", "curriculum", "subject", "subjects", "marks", "theory", "practical", "credits", "exam", "scheme"]
    if not any(k in text for k in keyword_hits):
        return None

    store = load_syllabus_store()
    if not store:
        return "I do not have the syllabus store yet. Please load the PDFs and generate `data/syllabus_store.json`."

    curriculum = detect_curriculum(text)
    branch = detect_branch(text)
    sem_range = detect_sem_range(text)
    sem_number = detect_sem_number(text)
    if sem_range is None and sem_number:
        if sem_number in (1, 2):
            sem_range = "1-2"
        elif sem_number in (3, 4):
            sem_range = "3-4"
        elif sem_number in (5, 6):
            sem_range = "5-6"
    if branch == "CS" and sem_range == "1-2":
        branch = "CSE"

    def get_entry(curr_key):
        if not curr_key or curr_key not in store:
            return None
        if not branch:
            return store[curr_key]
        branch_data = store[curr_key].get(branch)
        if not branch_data:
            return None
        if sem_range:
            return branch_data.get(sem_range)
        if len(branch_data) == 1:
            return next(iter(branch_data.values()))
        available = ", ".join(sorted(branch_data.keys()))
        return f"I have {branch} syllabus for semesters {available}. Please specify the semester range."

    selected_curriculum = curriculum
    entry = get_entry(curriculum) if curriculum else None
    if entry is None and curriculum is None:
        entry = get_entry("C20")
        if entry is not None:
            selected_curriculum = "C20"
        if entry is None:
            entry = get_entry("C25")
            if entry is not None:
                selected_curriculum = "C25"

    if not entry:
        return "I could not match the branch or semester. Please mention branch and semester (e.g., CSE 3-4 semesters)."

    if isinstance(entry, str):
        return entry

    if branch is None and isinstance(entry, dict):
        # entry is curriculum-level mapping: branch -> sem -> data
        if not sem_range:
            return "Please mention the semester (e.g., 3rd semester) so I can answer across branches."
        header = f"{(selected_curriculum or 'Curriculum').upper()} syllabus summary"
        wants_count = "how many subject" in text or "no of subject" in text or "number of subject" in text
        lines = [header]
        for branch_key, branch_data in sorted(entry.items()):
            data = branch_data.get(sem_range)
            if not data:
                continue
            text_blob = data.get("text", "")
            section = extract_sem_section(text_blob, sem_number)
            subject_lines = count_subject_lines(section)
            if wants_count and subject_lines:
                lines.append(f"{branch_key}: {len(subject_lines)} subjects")
            else:
                snippets = extract_relevant_snippets(section, message, limit=3)
                if snippets:
                    lines.append(f"{branch_key}: {snippets[0]}")
        if len(lines) > 1:
            return "\n".join(lines)
        return "I could not find matching syllabus data for that semester."

    text_blob = entry.get("text")
    if not text_blob:
        return "The syllabus text is missing for this selection."
    source = entry.get("label") or "Syllabus PDF"
    section = extract_sem_section(text_blob, sem_number)

    wants_count = "how many subject" in text or "no of subject" in text or "number of subject" in text
    wants_names = (
        "subject name" in text
        or "subject names" in text
        or ("subjects" in text or "subject" in text) and not wants_count
    )
    subject_lines = count_subject_lines(section)
    if wants_names:
        names = extract_subject_names(section)
        if not names:
            names = extract_course_titles(section, sem_number)
        if names:
            return f"{source}\n\nSubject names:\n" + "\n".join(names)

    if wants_count:
        if not subject_lines:
            subject_lines = extract_course_titles(section, sem_number)
        if subject_lines:
            count = len(subject_lines)
            sample = "\n".join(subject_lines[:8])
            return f"{source}\n\nTotal subjects found for this semester: {count}\n\nSample subjects:\n{sample}"

    snippets = extract_relevant_snippets(section, message)
    if snippets:
        snippet_text = "\n".join(snippets)
        return f"{source}\n\n{snippet_text}"

    trimmed = section[:1500].strip()
    if trimmed:
        return f"{source}\n\n{trimmed}"
    return "I could not find a matching section in the syllabus text."


@app.route("/")
def index():
    if session.get("student_roll") is None:
        return redirect(url_for("student_login"))
    return render_template(
        "index.html",
        college_name="SRI VENKATESHWARA POLYTECHNIC"
    )


@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"reply": "Please type a question so I can help."})

    syllabus_reply = find_syllabus_response(message)
    if syllabus_reply:
        log_chat(message, syllabus_reply, intent="syllabus", confidence=1.0)
        return jsonify({"reply": syllabus_reply, "intent": "syllabus", "confidence": 1.0})

    intent, confidence = predict_intent(message)
    if intent is None or confidence < CONFIDENCE_THRESHOLD:
        reply = "I am not fully sure. Could you please rephrase or ask about admissions, courses, fees, timetable, placements, or college info?"
        log_chat(message, reply, intent, confidence)
        return jsonify({"reply": reply, "intent": intent, "confidence": confidence})

    answer = fetch_answer(intent) or "I have information about this, but the detailed answer is not available in the database yet."
    log_chat(message, answer, intent, confidence)
    return jsonify({"reply": answer, "intent": intent, "confidence": confidence})


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, password_hash FROM admin_users WHERE username=%s", (username,))
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            if row and check_password_hash(row[1], password):
                session["admin_id"] = row[0]
                return redirect(url_for("admin_dashboard"))
        except Exception:
            pass
        flash("Invalid credentials", "error")
    return render_template("admin_login.html")


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_id", None)
    return redirect(url_for("admin_login"))


def require_admin():
    return session.get("admin_id") is not None


def require_student():
    return session.get("student_roll") is not None


@app.route("/student/login", methods=["GET", "POST"])
def student_login():
    if request.method == "POST":
        roll = request.form.get("roll", "").strip()
        password = request.form.get("password", "").strip()
        if not (roll and password):
            flash("Please enter your roll number and password", "error")
            return render_template("student_login.html")
        student = fetch_student_by_roll(roll)
        if not student:
            flash("Student not found. Please sign up first.", "error")
            return redirect(url_for("student_signup"))
        if not student[5]:
            session["pending_roll"] = roll
            flash("Please verify OTP to complete registration.", "error")
            return redirect(url_for("student_verify"))
        if not check_password_hash(student[4], password):
            flash("Invalid credentials", "error")
            return render_template("student_login.html")
        session["student_name"] = student[1]
        session["student_roll"] = student[2]
        return redirect(url_for("index"))
    return render_template("student_login.html")


@app.route("/student/signup", methods=["GET", "POST"])
def student_signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll = request.form.get("roll", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        if not (name and roll and email and password):
            flash("All fields are required", "error")
            return render_template("student_signup.html")

        existing_roll = fetch_student_by_roll(roll)
        existing_email = fetch_student_by_email(email)
        if existing_roll and existing_roll[5]:
            flash("Roll number already registered. Please login.", "error")
            return redirect(url_for("student_login"))
        if existing_email and existing_email[5]:
            flash("Email already registered. Please login.", "error")
            return redirect(url_for("student_login"))
        if existing_email and existing_email[2] != roll:
            flash("Email is already used with another roll number.", "error")
            return render_template("student_signup.html")

        otp_code = f"{random.randint(100000, 999999)}"
        otp_expires_at = datetime.now() + timedelta(minutes=10)
        password_hash = generate_password_hash(password)
        if existing_roll and not existing_roll[5]:
            created, err = update_student_pending_by_roll(name, roll, email, password_hash, otp_code, otp_expires_at)
        else:
            created, err = create_student(name, roll, email, password_hash, otp_code, otp_expires_at)
        if not created:
            flash(f"Signup failed: {err}", "error")
            return render_template("student_signup.html")

        session["pending_roll"] = roll
        sent, err = send_otp_email(email, name, otp_code)
        if not sent:
            flash(f"OTP send failed: {err}", "error")
            return render_template("student_signup.html")
        flash(f"OTP sent to {email}. Please check your inbox.", "success")
        return redirect(url_for("student_verify"))
    return render_template("student_signup.html")


@app.route("/student/verify", methods=["GET", "POST"])
def student_verify():
    pending_roll = session.get("pending_roll")
    if request.method == "POST":
        roll = request.form.get("roll", "").strip() or pending_roll
        otp = request.form.get("otp", "").strip()
        if not (roll and otp):
            flash("Please enter roll number and OTP", "error")
            return render_template("student_verify.html", roll=roll or "")
        student = fetch_student_by_roll(roll)
        if not student:
            flash("Student not found. Please sign up.", "error")
            return redirect(url_for("student_signup"))
        if student[5]:
            session["student_name"] = student[1]
            session["student_roll"] = student[2]
            session.pop("pending_roll", None)
            return redirect(url_for("index"))
        otp_code = student[6]
        otp_expires_at = student[7]
        if not otp_code or otp != otp_code:
            flash("Invalid OTP", "error")
            return render_template("student_verify.html", roll=roll)
        if otp_expires_at and datetime.now() > otp_expires_at:
            flash("OTP expired. Please sign up again to get a new OTP.", "error")
            return redirect(url_for("student_signup"))
        if not mark_student_verified(roll):
            flash("Verification failed. Please try again.", "error")
            return render_template("student_verify.html", roll=roll)
        session["student_name"] = student[1]
        session["student_roll"] = student[2]
        session.pop("pending_roll", None)
        return redirect(url_for("index"))
    return render_template("student_verify.html", roll=pending_roll or "")


@app.route("/student/profile")
def student_profile():
    if session.get("student_roll") is None:
        return redirect(url_for("student_login"))
    student = fetch_student_by_roll(session.get("student_roll"))
    if not student:
        return redirect(url_for("student_logout"))
    profile = {
        "name": student[1],
        "roll": student[2],
        "email": student[3],
        "verified_at": format_timestamp(student[8]) if student[8] else "Not verified",
    }
    return render_template("student_profile.html", profile=profile)


@app.route("/student/logout")
def student_logout():
    session.pop("student_name", None)
    session.pop("student_roll", None)
    return redirect(url_for("student_login"))


@app.route("/admin")
def admin_dashboard():
    if not require_admin():
        return redirect(url_for("admin_login"))

    qa_rows = []
    intents = []
    intent_dist, intent_last_updated = fetch_intent_distribution(limit=5)
    intent_trend = fetch_intent_trend(days=7)
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, question, answer, intent FROM qa_pairs ORDER BY id DESC")
        qa_rows = cursor.fetchall()
        cursor.execute("SELECT intent FROM intents ORDER BY intent")
        intents = [r[0] for r in cursor.fetchall()]
        cursor.close()
        conn.close()
    except Exception:
        pass

    return render_template(
        "admin_dashboard.html",
        qa_rows=qa_rows,
        intents=intents,
        intent_dist=intent_dist,
        intent_last_updated=intent_last_updated,
        intent_trend=intent_trend
    )


@app.route("/admin/add", methods=["POST"])
def admin_add():
    if not require_admin():
        return redirect(url_for("admin_login"))

    question = request.form.get("question", "").strip()
    answer = request.form.get("answer", "").strip()
    intent = request.form.get("intent", "").strip()

    if not (question and answer and intent):
        flash("All fields are required", "error")
        return redirect(url_for("admin_dashboard"))

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT IGNORE INTO intents (intent) VALUES (%s)", (intent,))
        cursor.execute(
            "INSERT INTO qa_pairs (question, answer, intent) VALUES (%s, %s, %s)",
            (question, answer, intent)
        )
        cursor.close()
        conn.close()
        flash("Q&A added successfully", "success")
    except Exception:
        flash("Failed to add Q&A", "error")

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/train", methods=["POST"])
def admin_train():
    if not require_admin():
        return redirect(url_for("admin_login"))

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT question, intent FROM qa_pairs")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            flash("No data to train", "error")
            return redirect(url_for("admin_dashboard"))

        import pandas as pd
        from train import train_model

        df = pd.DataFrame(rows, columns=["question", "intent"])
        temp_csv = os.path.join("data", "db_training.csv")
        df.to_csv(temp_csv, index=False)
        train_model(data_path=temp_csv, model_dir=MODEL_DIR, algo="logreg")
        flash("Model retrained successfully", "success")
    except Exception as e:
        flash(f"Training failed: {e}", "error")

    return redirect(url_for("admin_dashboard"))


if __name__ == "__main__":
    init_db()
    ensure_admin_user()
    app.run(debug=True)
