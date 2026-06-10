import re

def clean_text(text):
    # Remove page numbers (lines that are only digits, possibly with spaces)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if re.match(r'^\s*\d+\s*$', line.strip()):
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)
    
    # Remove repeated headers/footers (naive: remove identical consecutive lines)
    lines = text.split('\n')
    deduped = []
    prev = None
    for line in lines:
        if line.strip() != prev:
            deduped.append(line)
            prev = line.strip()
    text = '\n'.join(deduped)
    
    # Fix broken line wrapping: join lines that end with a lowercase letter or comma
    text = re.sub(r'(?<=[a-z,;:])\n(?=[a-z])', ' ', text)
    
    # Collapse multiple newlines to max 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' \n', '\n', text)
    
    return text.strip()

def clean_corpus(processed_dir="data/processed"):
    # Apply cleaning to all .txt files in processed_dir (recursive)
    import os
    from pathlib import Path
    for txt_file in Path(processed_dir).glob("**/*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            raw = f.read()
        cleaned = clean_text(raw)
        if cleaned != raw:
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(cleaned)
            print(f"Cleaned {txt_file}")
    print("Corpus cleaning complete.")

if __name__ == "__main__":
    clean_corpus()
