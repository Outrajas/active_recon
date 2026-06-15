# src/semantic_compiler/pipeline/segmentation.py
import PyPDF2
import re
import nltk

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

class SegmentationStage:
    """Stage 1: Extract text from PDF and split into non-overlapping sentence-based chunks."""
    def __init__(self, sentences_per_chunk: int = 5):
        self.sentences_per_chunk = sentences_per_chunk

    def run(self, pdf_path: str) -> list[str]:
        # Extract text using PyPDF2 (preserves line breaks)
        raw_text = ""
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    raw_text += page_text + "\n"

        # Normalize whitespace
        raw_text = re.sub(r'\n{3,}', '\n\n', raw_text)
        raw_text = re.sub(r'[ \t]+', ' ', raw_text)

        # Split into sentences
        sentences = nltk.sent_tokenize(raw_text)

        # Global deduplication: remove sentences that appear many times (headers, page numbers)
        from collections import Counter
        sent_counts = Counter(sentences)
        filtered_sentences = []
        for s in sentences:
            if sent_counts[s] > 2 and len(s) < 50:
                continue  # skip repeating short sentences (likely headers/page numbers)
            filtered_sentences.append(s)

        # Heading pattern to start new chunks
        heading_pattern = re.compile(
            r'^\s*(\d+\.?\s*\w+|Abstract|Introduction|Related Work|Conclusion|References|Acknowledgments|Appendix)\s*$',
            re.IGNORECASE
        )

        chunks = []
        current_chunk = []
        count = 0

        for i, sent in enumerate(filtered_sentences):
            stripped = sent.strip()
            is_heading = heading_pattern.match(stripped) is not None

            # If heading encountered and current chunk is not empty, finalize previous chunk
            if is_heading and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                count = 0

            current_chunk.append(stripped)
            count += 1

            # Split when we reach sentences_per_chunk and the next sentence is NOT a heading
            if count >= self.sentences_per_chunk:
                # Look ahead: if next sentence is a heading, defer split so heading starts its own chunk
                if i + 1 < len(filtered_sentences):
                    next_sent = filtered_sentences[i+1].strip()
                    if not heading_pattern.match(next_sent):
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        count = 0

        # Last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Filter out very short chunks
        return [c.strip() for c in chunks if len(c.strip()) > 20]
