#############################
# prompting.py
#############################

import json

# Whitelist única que usará el modelo (coincide con tus columnas category_*)
GPT_LABELS = [
    "Analytical, Diagnostic and Therapeutic Techniques and Equipment",
    "Anatomy",
    "Anthropology, Education, Sociology and Social Phenomena",
    "Chemicals and Drugs",
    "Disciplines and Occupations",
    "Diseases",
    "Health Care",
    "Humanities",
    "Information Science",
    "Named Groups",
    "Organisms",
    "Phenomena and Processes",
    "Psychiatry and Psychology",
    "Technology, Industry, Agriculture",
]


def _few_shot_block():
    return (
        "Examples:\n"
        "1) Text: \"Estudio del tratamiento de diabetes usando terapia con insulina\" -> "
        "[\"Diseases\", \"Chemicals and Drugs\", \"Analytical, Diagnostic and Therapeutic Techniques and Equipment\"]\n"
        "2) Text: \"Análisis de anatomía cerebral en pacientes neurológicos\" -> "
        "[\"Anatomy\", \"Diseases\"]\n"
        "3) Text: \"Efectos de metformina en pacientes geriátricos\" -> "
        "[\"Chemicals and Drugs\", \"Diseases\", \"Health Care\"]\n\n"
    )


def build_prompt(text, strategy):
    """
    Prompt simple: pide SOLO un array JSON de strings, sacado de la whitelist exacta.
    strategy: 'zero_shot' o 'few_shot' (añade ejemplos).
    """
    labels = json.dumps(GPT_LABELS, ensure_ascii=False)
    examples = _few_shot_block() if strategy == "few_shot" else ""
    return (
        "You are a multilabel medical text classifier.\n"
        "Task: map the Spanish input text to one or more categories from the EXACT whitelist below.\n"
        "RULES:\n"
        " - Output ONLY a JSON array of strings (no prose, no extra keys).\n"
        " - Use ONLY labels from the whitelist EXACTLY as written (in English).\n"
        " - If none apply, output []\n\n"
        f"Whitelist: {labels}\n\n"
        f"{examples}"
        "Text:\n"
        f"\"\"\"{text}\"\"\"\n\n"
        "JSON:"
    )
