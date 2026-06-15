# src/semantic_compiler/llm/canonicalize.py
import re

# Common academic stop‑phrases (section titles, generic terms)
SECTION_HEADERS = re.compile(
    r'^\d+(\.\d+)*\s+.*$|'                     # "1.", "3.2", "5.1 Training"
    r'^\d+\s*$|'                                # pure numbers
    r'^(abstract|introduction|related work|conclusion|references|acknowledg?ments|appendix)$',
    re.IGNORECASE
)

# Stop‑words that should never become canonical entities
STOP_WORDS = {
    "we","they","it","this","that","these","those",
    "another","each","every","all","both","few","many","most",
    "some","any","no","none","other","such","own",
    "day","days","score","part","section","figure","table",
    "one","two","three","first","second","third",
    "model","models","work","paper","result","results","approach",
    "method","methods","architecture","architectures",
    "=","+","-","*","/","pdk","qk","4to",
    "key","factor","way","ways","metric","metrics",
    "dimensionality","dimension","length","size","number"
}

def is_valid_entity(entity: str) -> bool:
    """Quick rejection of obvious non‑entities."""
    if len(entity) < 3:
        return False
    if entity in STOP_WORDS:
        return False
    if SECTION_HEADERS.match(entity.strip()):
        return False
    # Equation fragments: contain many mathematical symbols
    if re.search(r'[=+\-*/\^_\[\]{}()]', entity) and len(re.sub(r'\s+', '', entity)) < 10:
        return False
    return True

def canonical_name(entity: str) -> str:
    """Map surface forms to a canonical lower‑case term."""
    # Remove leading "the ", "a ", "an "
    cleaned = re.sub(r'^(the|a|an)\s+', '', entity.strip().lower())
    # Remove trailing punctuation
    cleaned = re.sub(r'[.,;:!?]+$', '', cleaned)
    # Normalise whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Common merges: "adam optimizer" -> "adam"
    merges = {
        "adam optimizer": "adam",
        "the transformer": "transformer",
        "transformer architecture": "transformer",
        "transformer model": "transformer",
        "bleu score": "bleu",
        "wmt 2014": "wmt",
    }
    return merges.get(cleaned, cleaned)
