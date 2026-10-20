# src/semantic_compiler/llm/canonicalize.py
import re

# SECTION_HEADERS now requires at least one dot (e.g., "3.2") to avoid matching "6 identical layers"
SECTION_HEADERS = re.compile(
    r'^\d+(\.\d+)+\s+.*$|'
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
    "dimensionality","dimension","length","size","number",
    # Precision filters for generic nouns
    "goal", "foundation", "set", "mean", "variance", "dot", "product",
    "same", "different", "another", "such", "way", "problem"
}

def is_valid_entity(entity: str) -> bool:
    # Debug (can be commented out in production)
    # print(f"[CANONICALIZE] Checking: '{entity}'")
    if len(entity) < 3:
        return False
    if entity in STOP_WORDS:
        return False
    if SECTION_HEADERS.match(entity.strip()):
        return False

    ent_lower = entity.strip().lower()

    # --- Layer count patterns (very permissive) ---
    if re.search(r'\b\d+\s*(identical\s+)?layers?\b', ent_lower):
        return True
    if re.search(r'n\s*=\s*\d+\s+layers?', ent_lower):
        return True
    if re.search(r'stack\s+of\s+(\d+|n\s*=\s*\d+)\s+layers?', ent_lower):
        return True
    if re.search(r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+sub-?layers?\b', ent_lower):
        return True
    # Hyphenated compounds
    if re.search(r'\w+-\w+', entity) and len(entity) > 4:
        return True
    # Known terms
    if entity.lower() in {'bleu', 'adam', 'dropout', 'label smoothing', 'warmup steps', 'feed-forward', 'self-attention', 'encoder-decoder', 'positional encoding'}:
        return True
    # Reject equation fragments only if short and math-heavy
    if re.search(r'[=+\-*/\^_\[\]{}()]', entity) and len(re.sub(r'\s+', '', entity)) < 10:
        return False
    # Single generic nouns not allowed unless in known list
    if len(entity.split()) == 1 and entity.lower() not in {'transformer', 'attention', 'encoder', 'decoder', 'layer', 'stack', 'network'}:
        return False
    return True

def canonical_name(entity: str) -> str:
    cleaned = entity.strip().lower()
    cleaned = re.sub(r'[.,;:!?]+$', '', cleaned)
    cleaned = re.sub(r'^(the|a|an)\s+', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    merges = {
        "adam optimizer": "adam",
        "the transformer": "transformer",
        "transformer architecture": "transformer",
        "transformer model": "transformer",
        "bleu score": "bleu",
        "wmt 2014": "wmt",
        "n=6 identical layers": "6 identical layers",
        "stack of n=6 identical layers": "stack of 6 identical layers",
    }
    return merges.get(cleaned, cleaned)
