
import os

DEFAULT_MODEL_NAME = os.environ.get("NLG_MODEL_NAME", "google/flan-t5-base")
FALLBACK_MODEL_NAME = "google/flan-t5-small"

MAX_PROMPT_LENGTH = 1000
DEFAULT_MAX_NEW_TOKENS = 150
DEFAULT_MIN_NEW_TOKENS = 10
DEFAULT_NUM_BEAMS = 4
DEFAULT_TEMPERATURE = 0.7
DEFAULT_REPETITION_PENALTY = 1.2
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3

SUPPORTED_STYLES = {
    "general": {
        "name": "General",
        "description": "Standard balanced natural language generation.",
        "prefix": "Generate a clear and natural paragraph incorporating all of the provided facts below. Do not leave out any numbers or percentages:"
    },
    "formal": {
        "name": "Formal",
        "description": "Professional, academic, and well-structured tone.",
        "prefix": "Write a formal and professional report incorporating all of the provided facts below. Do not leave out any numbers or percentages:"
    },
    "concise": {
        "name": "Concise",
        "description": "Brief, to-the-point summary of core details.",
        "prefix": "Summarize all of the provided facts below concisely in a short paragraph without leaving out any numbers or percentages:"
    },
    "descriptive": {
        "name": "Descriptive",
        "description": "Rich, detailed, and informative paragraph.",
        "prefix": "Write a comprehensive and detailed report explaining all of the provided facts below. Do not leave out any numbers or percentages:"
    }
}

MAX_HISTORY_ITEMS = 20
