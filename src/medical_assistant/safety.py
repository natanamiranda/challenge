
import re
import unicodedata


FORBIDDEN_EXPRESSIONS = [
    "sangue roxo",
    "osso marinho",
    "sensacao de tedio",
    "esgoto branco",
    "cerebra e spinal cord",
    "celulas de sangue roxo",
]

SENSITIVE_EXPRESSIONS = [
    "cura garantida",
    "cura e geralmente possivel",
    "a cura e dificil",
    "nao existe cura",
    "prognostico favoravel",
]

# Relação entre conceitos em português e expressões
# equivalentes encontradas nas fontes em inglês.
CLINICAL_TERM_GROUPS = {
    "cancer": ["cancer"],
    "leucemia": ["leukemia"],
    "medula ossea": ["bone marrow"],
    "linfoblastos": ["lymphoblasts", "lymphoblast"],
    "linfocitos": ["lymphocytes", "lymphocyte"],
    "febre": ["fever"],
    "cansaco": ["feeling tired", "tiredness", "tired", "fatigue"],
    "fraqueza": ["weakness"],
    "suores noturnos": ["night sweats"],
    "hematomas": ["bruising", "bruises"],
    "sangramento": ["bleeding"],
    "sangramentos": ["bleeding"],
    "petéquias": ["petechiae"],
    "falta de ar": ["shortness of breath"],
    "perda de peso": ["weight loss"],
    "perda de apetite": ["loss of appetite"],
    "dor nos ossos": ["pain in the bones"],
    "dor abdominal": ["pain in the stomach"],
    "sensacao de plenitude": ["feeling of fullness"],
    "carocos indolores": ["painless lumps"],
    "infeccoes frequentes": ["many infections", "frequent infections"],
    "anemia": ["anemia"],
    "hemorragia": ["bleeding", "hemorrhage"],
}


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def contains_expression(text: str, expression: str) -> bool:
    normalized_text = normalize_text(text)
    normalized_expression = normalize_text(expression)

    return normalized_expression in normalized_text


def validate_answer(
    answer: str,
    sources: list | None = None,
) -> dict:

    if not answer or not answer.strip():
        return {
            "approved": False,
            "requires_human_review": True,
            "lexical_check_passed": False,
            "reason": "A resposta está vazia.",
            "detected_issues": ["empty_answer"],
        }

    issues = []

    normalized_answer = normalize_text(answer)

    if not sources:
        issues.append("no_retrieved_sources")

    # Verifica expressões proibidas.
    for expression in FORBIDDEN_EXPRESSIONS:
        if contains_expression(normalized_answer, expression):
            issues.append(
                f"expressao_inadequada: {expression}"
            )

    # Verifica afirmações clínicas sensíveis.
    for expression in SENSITIVE_EXPRESSIONS:
        if contains_expression(normalized_answer, expression):
            issues.append(
                f"afirmacao_clinica_sensivel: {expression}"
            )

    # Verifica correspondência lexical dos conceitos
    # clínicos apresentados nas respostas com as fontes.
    if sources:
        source_text = " ".join(
            str(source.get("content", ""))
            for source in sources
        )

        normalized_sources = normalize_text(source_text)

        for portuguese_term, english_terms in CLINICAL_TERM_GROUPS.items():

            if not contains_expression(
                normalized_answer,
                portuguese_term,
            ):
                continue

            found_in_source = any(
                contains_expression(
                    normalized_sources,
                    english_term,
                )
                for english_term in english_terms
            )

            if not found_in_source:
                issues.append(
                    "termo_sem_correspondencia_na_fonte: "
                    f"{portuguese_term}"
                )

    # A verificação lexical não representa aprovação clínica.
    # Toda resposta continua exigindo revisão humana.
    return {
        "approved": False,
        "requires_human_review": True,
        "lexical_check_passed": len(issues) == 0,
        "reason": (
            "Verificação lexical executada. "
            "Ela não comprova a precisão semântica ou clínica. "
            "Aprovação humana obrigatória."
        ),
        "detected_issues": issues,
    }
