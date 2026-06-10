from .rewriter import Rewriter

_rewriter = None

def get_rewriter():
    global _rewriter
    if _rewriter is None:
        _rewriter = Rewriter()
    return _rewriter

def run_rewrite(text, style):
    rewriter = get_rewriter()
    return rewriter.rewrite(text, style)