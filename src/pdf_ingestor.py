import os
import fitz
import pdfplumber
import json
from pathlib import Path
from datetime import datetime

TRACKING_FILE = "outputs/processed_pdfs.json"

def load_processed():
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, "r") as f:
            return json.load(f)
    return {}

def save_processed(processed):
    os.makedirs(os.path.dirname(TRACKING_FILE), exist_ok=True)
    with open(TRACKING_FILE, "w") as f:
        json.dump(processed, f, indent=2)

def extract_text_pymupdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text if len(text.strip()) >= 100 else None
    except Exception as e:
        print(f"PyMuPDF error {pdf_path}: {e}")
        return None

def extract_text_pdfplumber(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
        return text if len(text.strip()) >= 100 else None
    except Exception as e:
        print(f"pdfplumber error {pdf_path}: {e}")
        return None

def process_new_pdfs(raw_dir="data/raw", processed_dir="data/processed"):
    styles = ["academic", "literary", "technical"]
    processed = load_processed()
    new_count = 0
    failed = []

    for style in styles:
        raw_style_dir = Path(raw_dir) / style
        if not raw_style_dir.exists():
            continue
        proc_style_dir = Path(processed_dir) / style
        proc_style_dir.mkdir(parents=True, exist_ok=True)

        for pdf_file in raw_style_dir.glob("*.pdf"):
            pdf_path_str = str(pdf_file.resolve())
            if pdf_path_str in processed:
                print(f"Skipping already processed: {pdf_file.name}")
                continue

            print(f"Processing new PDF: {pdf_file.name}")
            text = extract_text_pymupdf(pdf_file)
            if text is None:
                text = extract_text_pdfplumber(pdf_file)
            if text:
                output_txt = proc_style_dir / f"{pdf_file.stem}.txt"
                with open(output_txt, "w", encoding="utf-8") as f:
                    f.write(text)
                processed[pdf_path_str] = datetime.now().isoformat()
                new_count += 1
                print(f"Saved to {output_txt}")
            else:
                failed.append(pdf_path_str)
                print(f"Failed to extract from {pdf_file.name}")

    save_processed(processed)
    print(f"\nProcessed {new_count} new PDFs. Failed: {len(failed)}")
    if failed:
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/error_log.txt", "a") as err:
            err.write("PDF ingestion failures:\n" + "\n".join(failed) + "\n")

if __name__ == "__main__":
    process_new_pdfs()
