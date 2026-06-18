# src/semantic_compiler/llm/canonicalize.py
import re

SECTION_HEADERS = re.compile(
    r'^\d+(\.\d+)*\s+.*$|'
    r'^\d+\s*$|'
    r'^(abstract|introduction|related work|conclusion|references|acknowledg?ments|appendix)$',
    re.IGNORECASE
)

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
    # Debug: uncomment to see what's being checked
    # print(f"[CANONICALIZE] Checking: '{entity}'")
    if len(entity) < 3:
        return False
    if entity in STOP_WORDS:
        return False
    if SECTION_HEADERS.match(entity.strip()):
        return False
    # Allow layer count patterns (very permissive) – but we need to handle the string exactly as it comes
    ent_lower = entity.strip().lower()
    # Patterns for "6 identical layers", "N=6 layers", "stack of N=6 layers", "two sub-layers", etc.
    if re.match(r'^(\d+|n\s*=\s*\d+)\s+(identical\s+)?layers?$', ent_lower):
        return True
    if re.match(r'^stack\s+of\s+(\d+|n\s*=\s*\d+)\s+layers?$', ent_lower):
        return True
    if re.match(r'^stack\s+of\s+(\d+|n\s*=\s*\d+)\s+(identical\s+)?layers?$', ent_lower):
        return True
    if re.match(r'^(one|two|three|four|five|six|seven|eight|nine|ten)\s+sub-?layers?$', ent_lower):
        return True
    # Allow hyphenated compounds
    if re.search(r'\w+-\w+', entity) and len(entity) > 4:
        return True
    # Allow known terms
    if entity.lower() in {'bleu', 'adam', 'dropout', 'label smoothing', 'warmup steps', 'feed-forward', 'self-attention', 'encoder-decoder'}:
        return True
    # Reject equation fragments only if short and math-heavy
    if re.search(r'[=+\-*/\^_\[\]{}()]', entity) and len(re.sub(r'\s+', '', entity)) < 10:
        return False
    return True

def canonical_name(entity: str) -> str:
    cleaned = entity.strip().lower()
    cleaned = re.sub(r'[.,;:!?]+$', '', cleaned)
    cleaned = re.sub(r'^(the|a|an)\s+', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Only merge obvious aliases
    merges = {
        "adam optimizer": "adam",
        "the transformer": "transformer",
        "transformer architecture": "transformer",
        "transformer model": "transformer",
        "bleu score": "bleu",
        "wmt 2014": "wmt",
    }
    return merges.get(cleaned, cleaned)
