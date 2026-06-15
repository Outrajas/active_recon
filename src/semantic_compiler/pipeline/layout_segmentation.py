# src/semantic_compiler/pipeline/layout_segmentation.py
import pdfplumber
import re
from collections import defaultdict

class LayoutSegmentationStage:
    def __init__(self):
        pass

    def run(self, pdf_path: str) -> list[dict]:
        blocks = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                words = page.extract_words(keep_blank_chars=False, x_tolerance=2, y_tolerance=2)
                if not words:
                    continue
                lines = defaultdict(list)
                for w in words:
                    y_key = round(w['top'], 0)
                    lines[y_key].append(w)
                sorted_lines = sorted(lines.keys())
                current_block_lines = []
                current_block_type = None
                prev_y = None

                for y in sorted_lines:
                    line_words = sorted(lines[y], key=lambda w: w['x0'])
                    line_text = ' '.join(w['text'] for w in line_words).strip()
                    if not line_text:
                        continue

                    line_type = self._classify_line(line_text, line_words, prev_y, y, page_num)

                    if line_type != current_block_type or (prev_y is not None and (y - prev_y) > 15):
                        if current_block_lines:
                            blocks.append(self._create_block(current_block_lines, current_block_type, page_num))
                        current_block_lines = []
                        current_block_type = line_type
                    current_block_lines.append((y, line_text))
                    prev_y = y

                if current_block_lines:
                    blocks.append(self._create_block(current_block_lines, current_block_type, page_num))
        return blocks

    def _classify_line(self, line_text, line_words, prev_y, current_y, page_num):
        if re.match(r'^\s*\d+\s*$', line_text):
            return 'PAGE_NUMBER'
        if re.match(r'\s*\[\d+\]', line_text) or re.search(r'\b(arXiv|et\s+al\.|In\s+Proceedings|International\s+Conference|Neural\s+Information|Advances\s+in)\b', line_text, re.IGNORECASE):
            return 'REFERENCE'
        if page_num == 1 and len(line_text) > 10 and len(line_text) < 120 and not re.search(r'\d', line_text):
            return 'TITLE'
        if re.search(r'@\w+\.\w+', line_text) or re.search(r'Google\s(Brain|Research)|University\s+of\s+|Department\s+of\s+', line_text, re.IGNORECASE):
            return 'AUTHOR_BLOCK'
        if re.search(r'(\*|†|‡)|Equal\s+contribution|Work\s+performed|Currently\s+at|Now\s+at|This\s+work\s+was\s+done', line_text, re.IGNORECASE):
            return 'FOOTNOTE'
        if re.match(r'^\s*(Figure|Table)\s+\d+', line_text, re.IGNORECASE):
            return 'CAPTION'
        if re.match(r'^\s*(\d+(\.\d+)*\s+)?(Abstract|Introduction|Related\s+Work|Background|Method|Experiment|Results?|Discussion|Conclusion|References?|Acknowledgments?|Appendix|Supplementary)\s*$', line_text, re.IGNORECASE):
            return 'SECTION_HEADER'
        if re.search(r'\d+\s+\d+\.\d+', line_text) and len(re.findall(r'\d+', line_text)) > 3:
            return 'TABLE'
        math_chars = len(re.findall(r'[=+\-*/\\^_{}()\[\]]', line_text))
        if math_chars > 5 or (math_chars > 2 and len(line_text) < 80 and not re.search(r'[a-zA-Z]{3,}', line_text)):
            return 'EQUATION'
        return 'PARAGRAPH'

    def _create_block(self, lines, block_type, page_num):
        full_text = ' '.join(text for _, text in lines)
        avg_y = sum(y for y, _ in lines) / len(lines) if lines else 0
        return {
            'text': full_text,
            'document_role': block_type,
            'page': page_num,
            'y': round(avg_y, 1),
            'source_chunk': f"page{page_num}_{block_type}_{len(full_text)//100}"
        }
