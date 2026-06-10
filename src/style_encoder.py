AVAILABLE_STYLES = [
    "ACADEMIC",
    "LITERARY",
    "TECHNICAL",
    "MIXED"
]

def get_available_styles():
    return AVAILABLE_STYLES.copy()

def validate_style(style):
    return style.upper() in AVAILABLE_STYLES
