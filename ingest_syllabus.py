import os
import re
import json
import argparse
from datetime import datetime

from pypdf import PdfReader


DEFAULT_PDFS = [
    {"curriculum": "C20", "branch": "DAE", "sem": "1-2", "path": r"C:\Users\aishu\Downloads\C_20_DAE_1_2_Sem (1).pdf"},
    {"curriculum": "C20", "branch": "AT", "sem": "3-4", "path": r"C:\Users\aishu\Downloads\C_20_AT_3_4_Sem.pdf"},
    {"curriculum": "C20", "branch": "AT", "sem": "5-6", "path": r"C:\Users\aishu\Downloads\C_20_5_6_sem_AT.pdf"},
    {"curriculum": "C20", "branch": "CSE", "sem": "1-2", "path": r"C:\Users\aishu\Downloads\C_20_CSE_1_2_Sem (1).pdf"},
    {"curriculum": "C20", "branch": "CS", "sem": "3-4", "path": r"C:\Users\aishu\Downloads\C_20_CS_3_4_Sem.pdf"},
    {"curriculum": "C20", "branch": "CSE", "sem": "5-6", "path": r"C:\Users\aishu\Downloads\C20_5_6_sem_CSE (1).pdf"},
    {"curriculum": "C20", "branch": "ECE", "sem": "1-2", "path": r"C:\Users\aishu\Downloads\C_20_ECE_1_2_Sem.pdf"},
    {"curriculum": "C20", "branch": "EC", "sem": "3-4", "path": r"C:\Users\aishu\Downloads\C_20_EC_3_4_Sem.pdf"},
    {"curriculum": "C20", "branch": "ECE", "sem": "5-6", "path": r"C:\Users\aishu\Downloads\C20_5_6_sem_ECE.pdf"},
    {"curriculum": "C20", "branch": "DME", "sem": "1-2", "path": r"C:\Users\aishu\Downloads\C_20_DME_1_2_Sem.pdf"},
    {"curriculum": "C20", "branch": "ME", "sem": "3-4", "path": r"C:\Users\aishu\Downloads\C_20_ME_3_4_Sem.pdf"},
    {"curriculum": "C20", "branch": "ME", "sem": "5-6", "path": r"C:\Users\aishu\Downloads\C20_5_6_sem_ME.pdf"},
    {"curriculum": "C25", "branch": "CSE", "sem": "1-4", "path": r"C:\Users\aishu\Downloads\C_25_Draft_1_4_ComputerScience&Engineering (1).pdf"},
    {"curriculum": "C25", "branch": "EC", "sem": "1-2", "path": r"C:\Users\aishu\Downloads\23_C_25_EC_1_2_Sem_Electronics_and_Communication_Engg.pdf"},
    {"curriculum": "C25", "branch": "AT", "sem": "1-4", "path": r"C:\Users\aishu\Downloads\C_25_Draft_AT_1_4_AutomobileEngineering.pdf"},
    {"curriculum": "C25", "branch": "CSE", "sem": "1-4", "path": r"C:\Users\aishu\Downloads\C_25_Draft_1_4_ComputerScience&Engineering.pdf"},
]


def normalize_text(text: str):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(path: str):
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return normalize_text("\n".join(parts))


def resolve_path(item, root_dir):
    if os.path.exists(item["path"]):
        return item["path"]
    if root_dir:
        candidate = os.path.join(root_dir, os.path.basename(item["path"]))
        if os.path.exists(candidate):
            return candidate
    return None


def build_store(entries, root_dir):
    store = {}
    for item in entries:
        resolved = resolve_path(item, root_dir)
        if not resolved:
            print(f"Missing file: {item['path']}")
            continue
        text = extract_pdf_text(resolved)
        label = f"{item['curriculum']} {item['branch']} Sem {item['sem']}"
        store.setdefault(item["curriculum"], {}).setdefault(item["branch"], {})[item["sem"]] = {
            "label": label,
            "source_path": resolved,
            "text": text,
            "extracted_at": datetime.now().isoformat(timespec="seconds")
        }
        print(f"Loaded: {label} ({len(text)} chars)")
    return store


def main():
    parser = argparse.ArgumentParser(description="Extract syllabus PDFs into a JSON store.")
    parser.add_argument("--root", help="Optional folder to resolve missing PDF files.")
    parser.add_argument("--out", default=os.path.join("data", "syllabus_store.json"), help="Output JSON path.")
    args = parser.parse_args()

    store = build_store(DEFAULT_PDFS, args.root)
    if not store:
        print("No PDFs were loaded. Please check file paths.")
        return 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
    print(f"Saved store to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
