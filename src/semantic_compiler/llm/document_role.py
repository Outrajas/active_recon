# src/semantic_compiler/llm/document_role.py
import re

class DocumentRoleClassifier:
    BIB_PATTERNS = [
        r'\b(arXiv|et\s+al\.|In\s+Proceedings|International\s+Conference|Neural\s+Information|Advances\s+in)',
        r'\.\s*\d{4}\.?\s*\[',
        r'\[\d+\]',
        r'^\s*\[\d+\]',
        r'http',
        r'doi\s*:',
    ]
    AUTHOR_PATTERNS = [
        r'@\w+\.\w+',
        r'Google\s(Brain|Research)',
        r'University\s+of\s+',
        r'Department\s+of\s+',
    ]
    FOOTNOTE_PATTERNS = [
        r'^\s*[*†‡]',
        r'Equal\s+contribution',
        r'Work\s+performed\s+(while|at|during)',
        r'Currently\s+at',
        r'Now\s+at',
        r'This\s+work\s+was\s+done',
    ]
    SECTION_HEADER_PATTERN = re.compile(
        r'^\s*(\d+(\.\d+)*\s+)?(Abstract|Introduction|Related\s+Work|Background|Method|Experiment|Results?|Discussion|Conclusion|References?|Acknowledgments?|Appendix|Supplementary)\s*$',
        re.IGNORECASE
    )
    CAPTION_START = re.compile(r'^\s*(Figure|Table)\s+\d+', re.IGNORECASE)

    def classify(self, sentence: str) -> str:
        if not sentence or len(sentence.strip()) < 3:
            return "METADATA"

        # Only if the sentence starts with "Figure/Table" do we consider it a caption
        if self.CAPTION_START.match(sentence):
            return "CAPTION"

        # Section header
        if self.SECTION_HEADER_PATTERN.match(sentence.strip()):
            return "SECTION_HEADER"

        # Bibliography
        bib_score = sum(1 for p in self.BIB_PATTERNS if re.search(p, sentence, re.IGNORECASE))
        if bib_score >= 2 or (bib_score >= 1 and len(re.findall(r'\[\d+\]', sentence)) > 3):
            return "BIBLIOGRAPHY"

        # Author block
        if any(re.search(p, sentence, re.IGNORECASE) for p in self.AUTHOR_PATTERNS):
            return "AUTHOR_BLOCK"
        # Footnote
        if any(re.search(p, sentence, re.IGNORECASE) for p in self.FOOTNOTE_PATTERNS):
            return "FOOTNOTE"

        # Everything else is MAIN_CONTENT (equations/tables will be extracted as part of text, not filtered)
        return "MAIN_CONTENT"
