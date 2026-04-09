import os
import pandas as pd
import mysql.connector

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "Root@123")
DB_NAME = os.environ.get("DB_NAME", "college_chatbot")


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


def seed_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    if df.empty:
        print("No data found in CSV")
        return

    conn = get_db_connection()
    cursor = conn.cursor()

    for _, row in df.iterrows():
        question = str(row["question"]).strip()
        intent = str(row["intent"]).strip()
        answer = str(row["answer"]).strip() if "answer" in df.columns else ""
        if not (question and intent and answer):
            continue
        cursor.execute("INSERT IGNORE INTO intents (intent) VALUES (%s)", (intent,))
        cursor.execute(
            "INSERT INTO qa_pairs (question, answer, intent) VALUES (%s, %s, %s)",
            (question, answer, intent)
        )

    cursor.close()
    conn.close()
    print("Seeded DB successfully")


if __name__ == "__main__":
    seed_from_csv(os.path.join("data", "intents.csv"))
