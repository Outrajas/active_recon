# src/semantic_compiler/llm/concept_extractor.py
import spacy, re

KNOWN_CONCEPTS = {
    "transformer", "self-attention", "multi-head attention",
    "encoder", "decoder", "positional encoding",
    "residual connection", "layer normalization", "feed-forward",
    "attention mechanism", "attention mechanisms",
    "recurrence", "convolution", "parallelization",
    "bleu", "perplexity", "accuracy", "f1", "loss",
    "wmt", "wmt14", "en-de", "en-fr",
    "adam", "sgd", "dropout", "label smoothing",
    "beam search", "bpe", "word-piece",
    "bytenet", "convs2s", "gnmt", "mo", "deep-att",
    "tensor2tensor", "tpu", "gpu", "nmt",
    "sequence transduction", "machine translation",
    "softmax", "relu", "gelu", "layernorm",
    "query", "key", "value", "scaled dot-product attention",
    "embedding", "embedding layer",
}

DISCOURSE_BLACKLIST = {
    "our model", "our approach", "our method", "our architecture",
    "the model", "the approach", "the method", "the architecture",
    "this model", "this approach", "this method", "this architecture",
    "their model", "their approach",
}

class ConceptExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_concepts(self, text: str) -> list:
        doc = self.nlp(text)
        concepts = set()

        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "GPE", "WORK_OF_ART"):
                clean = ent.text.strip().lower()
                if len(clean) > 2 and not self._is_stopword(clean) and clean not in DISCOURSE_BLACKLIST:
                    concepts.add(clean)

        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip().lower()
            if chunk_text in DISCOURSE_BLACKLIST:
                continue
            if re.search(r'[A-Z]', chunk.text) and len(chunk_text.split()) <= 4:
                if self._is_valid_concept(chunk_text):
                    concepts.add(chunk_text)

        for known in KNOWN_CONCEPTS:
            if re.search(r'\b' + re.escape(known) + r'\b', text, re.IGNORECASE):
                concepts.add(known)

        concepts = {c for c in concepts if not re.match(r'^(figure|table|section)\s*\d+', c)}
        concepts = {c for c in concepts if c not in DISCOURSE_BLACKLIST and self._is_valid_concept(c)}
        return sorted(concepts)

    def _is_valid_concept(self, concept: str) -> bool:
        if len(concept) < 3: return False
        if not any(c.isalpha() for c in concept): return False
        stopwords = {"the","a","an","this","that","these","those","each","every","all","some","any","no","other","such","own","we","they","it","its","our","your","my","his","her","their","for","but","and","or","not","with","without","from","into"}
        if concept.lower() in stopwords: return False
        if len(concept) <= 2 and concept not in KNOWN_CONCEPTS: return False
        non_alpha = sum(1 for ch in concept if not ch.isalpha())
        if non_alpha > len(concept) * 0.5: return False
        return True

    def _is_stopword(self, word: str) -> bool:
        stopwords = {"the","a","an","this","that","these","those","each","every","all","some","any","no","other","such","own","we","they","it","its","our","your","my","his","her","their"}
        return word.lower() in stopwords
