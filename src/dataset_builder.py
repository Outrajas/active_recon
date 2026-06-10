import os
import json
import re
from pathlib import Path

DATASET_FILE = "data/datasets/style_continuation.jsonl"
PROCESSED_DIR = "data/processed"

# Patterns to detect metadata lines
METADATA_PATTERNS = [
    r"Proceedings of",
    r"Association for Computational Linguistics",
    r"@\w+\.com",
    r"ISBN",
    r"pages \d+",
    r"Volume",
    r"Conference on",
    r"https?://",
    r"copyright",
    r"©",
    r"Submitted to",
    r"Received",
    r"Accepted",
    r"Published in",
    r"arXiv",
]

def is_metadata_line(line):
    line_lower = line.lower()
    for pat in METADATA_PATTERNS:
        if re.search(pat, line_lower):
            return True
    return False

def clean_text(text, skip_first_words=1000):
    lines = text.split('\n')
    filtered_lines = [line for line in lines if not is_metadata_line(line)]
    text = '\n'.join(filtered_lines)
    if skip_first_words > 0:
        words = text.split()
        if len(words) > skip_first_words:
            text = ' '.join(words[skip_first_words:])
    return text

def chunk_text(text, min_words=150, target_words=300, max_words=450):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        word_count = len(sent.split())
        # If a single sentence exceeds max_words, split it artificially
        if word_count > max_words:
            words = sent.split()
            for i in range(0, word_count, max_words):
                piece = ' '.join(words[i:i+max_words])
                if len(piece.split()) >= min_words:
                    chunks.append(piece)
            continue
        # Otherwise, accumulate sentences normally
        # (we'll rebuild the logic to use a proper accumulator)
    # Re-implement accumulation because the above loop splits chunks differently
    # Better: collect all sentences (split long ones) then accumulate
    all_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        wc = len(sent.split())
        if wc > max_words:
            words = sent.split()
            for i in range(0, wc, max_words):
                piece = ' '.join(words[i:i+max_words])
                all_sentences.append(piece)
        else:
            all_sentences.append(sent)
    # Now accumulate into chunks
    chunks = []
    current = []
    cur_len = 0
    for sent in all_sentences:
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

def split_chunk_into_prompt_target(chunk, split_ratio=0.7):
    words = chunk.split()
    split_idx = max(1, int(len(words) * split_ratio))
    # Find sentence boundary
    best_idx = split_idx
    search_range = min(30, len(words) - split_idx)
    for i in range(split_idx, min(split_idx + search_range, len(words))):
        if words[i-1].endswith(('.', '!', '?')):
            best_idx = i
            break
    prompt_words = words[:best_idx]
    target_words = words[best_idx:]
    if len(target_words) < 15:
        target_start = max(1, int(len(words) * 0.7))
        prompt_words = words[:target_start]
        target_words = words[target_start:]
    return ' '.join(prompt_words), ' '.join(target_words)

def build_dataset():
    styles = ["academic", "literary", "technical"]
    all_records = []
    total_chunks_processed = 0
    total_pairs_generated = 0
    prompt_lengths = []
    target_lengths = []

    for style in styles:
        style_dir = Path(PROCESSED_DIR) / style
        if not style_dir.exists():
            print(f"Warning: {style_dir} does not exist, skipping {style}")
            continue

        for txt_file in style_dir.glob("*.txt"):
            print(f"Processing {txt_file}")
            with open(txt_file, "r", encoding="utf-8") as f:
                raw_text = f.read()
            cleaned = clean_text(raw_text, skip_first_words=1000)
            if len(cleaned.split()) < 300:
                print(f"  -> Skipping {txt_file.name} – too short after cleaning")
                continue
            chunks = chunk_text(cleaned, min_words=150, target_words=300, max_words=450)
            total_chunks_processed += len(chunks)
            for chunk in chunks:
                if len(chunk.split()) < 100:
                    continue
                prompt_text, target_text = split_chunk_into_prompt_target(chunk, split_ratio=0.7)
                if len(target_text.split()) < 20:
                    continue
                record = {
                    "style": style.upper(),
                    "prompt": prompt_text,
                    "target": target_text
                }
                all_records.append(record)
                total_pairs_generated += 1
                prompt_lengths.append(len(prompt_text.split()))
                target_lengths.append(len(target_text.split()))

    os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    if total_pairs_generated > 0:
        avg_prompt = sum(prompt_lengths) / len(prompt_lengths)
        avg_target = sum(target_lengths) / len(target_lengths)
        min_prompt = min(prompt_lengths)
        max_prompt = max(prompt_lengths)
        min_target = min(target_lengths)
        max_target = max(target_lengths)
    else:
        avg_prompt = avg_target = min_prompt = max_prompt = min_target = max_target = 0

    print(f"\nDataset summary:")
    print(f"  Total pairs: {total_pairs_generated}")
    print(f"  Processed chunks: {total_chunks_processed}")
    print(f"  Prompt word count – avg: {avg_prompt:.1f}, min: {min_prompt}, max: {max_prompt}")
    print(f"  Target word count – avg: {avg_target:.1f}, min: {min_target}, max: {max_target}")
    print(f"  Saved to {DATASET_FILE}")

if __name__ == "__main__":
    build_dataset()