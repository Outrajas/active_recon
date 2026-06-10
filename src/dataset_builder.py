import os
import json
import random
import re
from pathlib import Path
from datetime import datetime

DATASET_FILE = "data/datasets/style_corrupt_pairs.jsonl"
PROCESSED_PDFS_FILE = "outputs/processed_pdfs.json"
CHUNK_TRACKING_FILE = "outputs/chunked_files.json"

def load_processed_pdfs():
    if os.path.exists(PROCESSED_PDFS_FILE):
        with open(PROCESSED_PDFS_FILE, "r") as f:
            return json.load(f)
    return {}

def load_chunked_files():
    if os.path.exists(CHUNK_TRACKING_FILE):
        with open(CHUNK_TRACKING_FILE, "r") as f:
            return json.load(f)
    return {}

def save_chunked_files(chunked):
    os.makedirs(os.path.dirname(CHUNK_TRACKING_FILE), exist_ok=True)
    with open(CHUNK_TRACKING_FILE, "w") as f:
        json.dump(chunked, f, indent=2)

def corrupt_text(text, corruption_prob=0.25):
    words = text.split()
    if len(words) < 8:
        return text
    corrupted = [w for w in words if random.random() > corruption_prob]
    if len(corrupted) < 4:
        corrupted = words[:4]
    if len(corrupted) > 3:
        start = random.randint(0, len(corrupted)-3)
        end = random.randint(start+2, min(start+5, len(corrupted)))
        block = corrupted[start:end]
        random.shuffle(block)
        corrupted[start:end] = block
    if random.random() < 0.3:
        corrupted = [re.sub(r'[^\w\s]', '', w) for w in corrupted]
    if random.random() < 0.2 and '.' in text:
        sentences = text.split('. ')
        if len(sentences) > 1:
            keep = random.randint(1, len(sentences)-1)
            corrupted = sentences[:keep]
            corrupted = ' '.join(corrupted).split()
    return ' '.join(corrupted)

def chunk_text(text, min_words=60, target_words=120, max_words=200):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    cur_len = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        wc = len(sent.split())
        if cur_len + wc <= target_words:
            current.append(sent)
            cur_len += wc
        else:
            if cur_len >= min_words:
                chunks.append(' '.join(current))
            current = [sent]
            cur_len = wc
    if current and cur_len >= min_words:
        chunks.append(' '.join(current))
    return chunks

def build_dataset_incremental(processed_dir="data/processed"):
    processed_pdfs = load_processed_pdfs()
    chunked_files = load_chunked_files()
    os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)
    mode = "a" if os.path.exists(DATASET_FILE) else "w"
    total_new = 0
    with open(DATASET_FILE, mode, encoding="utf-8") as out_f:
        for pdf_path, _ in processed_pdfs.items():
            if pdf_path in chunked_files:
                continue
            # Build correct processed txt path: data/processed/{style}/{filename}.txt
            pdf_path_obj = Path(pdf_path)
            style = pdf_path_obj.parent.name  # academic, literary, technical
            txt_filename = pdf_path_obj.stem + ".txt"
            txt_path = Path(processed_dir) / style / txt_filename
            if not txt_path.exists():
                print(f"Processed text missing: {txt_path}")
                continue
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            style_upper = style.upper()
            chunks = chunk_text(text)
            for chunk in chunks:
                if len(chunk.split()) < 30:
                    continue
                corrupted = corrupt_text(chunk)
                if corrupted == chunk or len(corrupted.split()) < 10:
                    words = chunk.split()
                    drop = random.sample(range(len(words)), min(3, len(words)//5))
                    corrupted = ' '.join(w for i,w in enumerate(words) if i not in drop)
                record = {"style": style_upper, "corrupted": corrupted, "original": chunk}
                out_f.write(json.dumps(record) + "\n")
                total_new += 1
            chunked_files[pdf_path] = datetime.now().isoformat()
    save_chunked_files(chunked_files)
    print(f"Added {total_new} new corrupt-original pairs to dataset.")

if __name__ == "__main__":
    build_dataset_incremental()
