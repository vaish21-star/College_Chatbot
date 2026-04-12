import os
import re
import json
import argparse
from datetime import datetime

from dotenv import load_dotenv
import mysql.connector

from ingest_syllabus import DEFAULT_PDFS, build_store


BRANCH_NAMES = {
    "CSE": "Computer Science Engineering",
    "CS": "Computer Science",
    "ECE": "Electronics and Communication Engineering",
    "EC": "Electronics and Communication",
    "DME": "Mechanical Engineering (Diploma)",
    "ME": "Mechanical Engineering",
    "AT": "Automobile Engineering",
    "DAE": "Automobile Engineering (Diploma)",
}


def get_db_connection():
    return mysql.connector.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", "Root@123"),
        database=os.environ.get("DB_NAME", "college_chatbot"),
        autocommit=True,
    )


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


def extract_course_titles(text: str):
    titles = []
    pattern = re.compile(
        r"Course Title\s*[:\-]?\s*(.+?)(?:Category|No\.\s*of\s*Credits|Type\s*of\s*Course|CIE\s*Marks|SEE\s*Marks|Total\s*Contact\s*Hours|Teaching\s*Scheme|$)",
        re.IGNORECASE | re.DOTALL
    )
    for m in pattern.finditer(text):
        raw = m.group(1)
        raw = re.sub(r"\s+", " ", raw).strip()
        raw = re.sub(r"\s*Course Group.*$", "", raw, flags=re.IGNORECASE).strip()
        if raw:
            titles.append(raw)
    seen = set()
    unique = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def ensure_intent(cursor, intent):
    cursor.execute("INSERT IGNORE INTO intents (intent) VALUES (%s)", (intent,))


def qa_exists(cursor, question, intent):
    cursor.execute("SELECT COUNT(*) FROM qa_pairs WHERE question=%s AND intent=%s", (question, intent))
    return cursor.fetchone()[0] > 0


def insert_qa(cursor, question, answer, intent):
    ensure_intent(cursor, intent)
    if qa_exists(cursor, question, intent):
        return False
    cursor.execute(
        "INSERT INTO qa_pairs (question, answer, intent) VALUES (%s,%s,%s)",
        (question, answer, intent)
    )
    return True


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Seed Q&A from syllabus PDFs.")
    parser.add_argument("--root", help="Optional folder to resolve missing PDF files.")
    parser.add_argument("--out", default=os.path.join("data", "syllabus_store.json"))
    args = parser.parse_args()

    store = None
    if os.path.exists(args.out):
        try:
            with open(args.out, "r", encoding="utf-8") as f:
                store = json.load(f)
        except Exception:
            store = None

    if not store:
        store = build_store(DEFAULT_PDFS, args.root)
        if not store:
            print("No PDFs loaded. Check PDF paths.")
            return 1
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)

    conn = get_db_connection()
    cursor = conn.cursor()

    branches = sorted({b for c in store.values() for b in c.keys()})
    courses_list = []
    for b in branches:
        full = BRANCH_NAMES.get(b, b)
        courses_list.append(f"{b} - {full}")
    courses_answer = "Available courses/branches:\n" + "\n".join(courses_list)
    insert_qa(
        cursor,
        "What courses are available?",
        courses_answer,
        "courses_available"
    )

    inserted = 0
    for curriculum, branches_data in store.items():
        for branch, sems in branches_data.items():
            for sem, data in sems.items():
                text = data.get("text", "")
                names = extract_subject_names(text)
                if not names:
                    names = extract_course_titles(text)
                if not names:
                    continue
                intent = f"subjects_{curriculum.lower()}_{branch.lower()}_{sem.replace('-', '_')}"
                question = f"List subject names for {curriculum} {branch} {sem} semesters."
                answer = f"{curriculum} {branch} Sem {sem} subject names:\n" + "\n".join(names)
                if insert_qa(cursor, question, answer, intent):
                    inserted += 1

    cursor.close()
    conn.close()
    print(f"Seeded Q&A entries: {inserted + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
