
import json
import re
import unicodedata
from pathlib import Path

from langchain_ollama import ChatOllama

from medical_assistant.config import settings
from medical_assistant.rag import search_protocols
from medical_assistant.safety import validate_answer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GLOSSARY_FILE = (
    PROJECT_ROOT / "data" / "protocols" / "medical_glossary.json"
)


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    return re.sub(r"\s+", " ", text).strip()


def load_medical_glossary():
    if not GLOSSARY_FILE.exists():
        return {}

    with GLOSSARY_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_llm():
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        num_predict=350,
    )


def translate_symptoms_from_source(source_text: str):
    source = normalize_text(source_text)

    translations = [
        (["fever"], "Febre"),
        (
            ["feeling tired", "weakness"],
            "Cansaço ou fraqueza",
        ),
        (["night sweats"], "Suores noturnos"),
        (
            ["easy bruising or bleeding"],
            "Hematomas ou sangramentos fáceis",
        ),
        (
            ["petechiae"],
            "Petéquias (pequenos pontos na pele causados por sangramento)",
        ),
        (
            ["shortness of breath"],
            "Falta de ar",
        ),
        (["weight loss"], "Perda de peso"),
        (["loss of appetite"], "Perda de apetite"),
        (
            ["pain in the bones or stomach"],
            "Dor nos ossos ou no abdômen",
        ),
        (
            ["pain or feeling of fullness below the ribs"],
            "Dor ou sensação de plenitude abaixo das costelas",
        ),
        (
            ["painless lumps in the neck, underarm, stomach, or groin"],
            "Caroços indolores no pescoço, axilas, abdômen ou virilha",
        ),
        (
            ["having many infections"],
            "Infecções frequentes",
        ),
    ]

    symptoms = []

    for expressions, translation in translations:
        if any(
            normalize_text(expression) in source
            for expression in expressions
        ):
            symptoms.append(translation)

    if not symptoms:
        return None

    answer = (
        "Segundo a fonte médica consultada, os sinais e sintomas "
        "da leucemia linfoblástica aguda em adultos podem incluir:\n\n"
    )

    answer += "\n".join(f"- {symptom}" for symptom in symptoms)

    answer += (
        "\n\nA fonte ressalta que esses sinais também podem "
        "ser causados por outras condições. A presença desses "
        "sintomas, isoladamente, não confirma um diagnóstico."
    )

    return answer


def translate_definition_from_source(source_text: str):
    source = normalize_text(source_text)

    translations = [
        (
            "become lymphoblasts, b lymphocytes, or t lymphocytes",
            "As células envolvidas podem se tornar linfoblastos "
            "ou linfócitos dos tipos B ou T.",
        ),
        (
            "these cells are also called leukemia cells",
            "Essas células também são chamadas de células leucêmicas.",
        ),
        (
            "not able to fight infection very well",
            "Essas células não conseguem combater infecções "
            "adequadamente.",
        ),
        (
            "there is less room for healthy white blood cells, "
            "red blood cells, and platelets",
            "O aumento dessas células no sangue e na medula óssea "
            "reduz o espaço disponível para células brancas saudáveis, "
            "glóbulos vermelhos e plaquetas.",
        ),
        (
            "this may cause infection, anemia, and easy bleeding",
            "Essa alteração pode favorecer infecções, anemia "
            "e sangramentos.",
        ),
        (
            "the cancer can also spread to the central nervous system",
            "A doença também pode se espalhar para o sistema "
            "nervoso central.",
        ),
    ]

    translated_parts = []

    for expression, translation in translations:
        if normalize_text(expression) in source:
            translated_parts.append(translation)

    if not translated_parts:
        return None

    answer = (
        "A leucemia linfoblástica aguda (LLA) é um tipo de câncer "
        "do sangue e da medula óssea.\n\n"
        "O trecho recuperado da fonte médica explica que:\n\n"
    )

    answer += "\n".join(
        f"- {part}"
        for part in translated_parts
    )

    is_part_three = bool(
        re.search(r"part\s*3\s*of\s*3", source)
    )

    if is_part_three:
        answer += (
            "\n\n**Limitação da fonte:** foi recuperada apenas "
            "a terceira parte de um documento com três partes. "
            "Portanto, esta resposta apresenta somente as informações "
            "disponíveis nesse trecho, não uma definição completa."
        )

    return answer


def generate_llm_answer(question, documents, patient_context=None):
    glossary = load_medical_glossary()

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    glossary_context = "\n".join(
        f"- {english}: {portuguese}"
        for english, portuguese in glossary.items()
    )

    patient_information = (
        json.dumps(
            patient_context,
            ensure_ascii=False,
            indent=2,
        )
        if patient_context
        else "Nenhum contexto de paciente foi fornecido."
    )

    prompt = f"""
Você é um assistente de informação médica.

Gere um rascunho em português brasileiro utilizando
exclusivamente os documentos fornecidos.

REGRAS:
- Responda exclusivamente em português brasileiro.
- Não apresente frases ou parágrafos em inglês.
- Traduza o conteúdo médico dos documentos para português.
- Preserve nomes próprios, siglas e termos técnicos reconhecidos quando necessário.
- Não invente informações clínicas.
- Não acrescente sintomas ausentes das fontes.
- Não faça diagnósticos.
- Não prescreva medicamentos ou tratamentos.
- Não traduza termos médicos por aproximação.
- Não invente nomes de doenças.
- Não acrescente recomendações clínicas ausentes das fontes.
- Responda somente ao que foi perguntado.
- Informe em português quando os documentos forem insuficientes.
- Não utilize o contexto do paciente como fonte de conhecimento médico.

GLOSSÁRIO:
{glossary_context}

CONTEXTO FICTÍCIO DO PACIENTE:
{patient_information}

DOCUMENTOS:
{context}

PERGUNTA:
{question}

RESPOSTA EM PORTUGUÊS:
"""

    response = create_llm().invoke(prompt)

    return str(response.content).strip()


def generate_answer(
    question: str,
    patient_context: dict | None = None,
):
    if not question or not question.strip():
        raise ValueError("A pergunta não pode estar vazia.")

    documents = search_protocols(question, k=3)

    if not documents:
        return {
            "answer": (
                "Não foram encontradas fontes suficientes "
                "para responder à pergunta."
            ),
            "sources": [],
            "safety": {
                "approved": False,
                "requires_human_review": True,
                "reason": "Nenhuma fonte foi recuperada.",
                "detected_issues": ["no_retrieved_sources"],
                "lexical_check_passed": False,
                "human_review_status": "pending",
            },
        }

    sources = [
        {
            "source": document.metadata.get("source"),
            "url": document.metadata.get("source_url"),
        }
        for document in documents
    ]

    question_normalized = normalize_text(question)

    is_symptom_question = any(
        term in question_normalized
        for term in [
            "sintoma",
            "sintomas",
            "sinais",
            "symptom",
            "symptoms",
        ]
    )

    is_definition_question = any(
        term in question_normalized
        for term in [
            "o que e",
            "definicao",
            "explique",
            "what is",
            "define",
            "definition",
            "explain",
        ]
    )

    answer = None

    if is_symptom_question:
        for document in documents:
            answer = translate_symptoms_from_source(
                document.page_content
            )

            if answer:
                break

    elif is_definition_question:
        for document in documents:
            answer = translate_definition_from_source(
                document.page_content
            )

            if answer:
                break

    if not answer:
        answer = generate_llm_answer(
            question=question,
            documents=documents,
            patient_context=patient_context,
        )

    validation_sources = [
        {"content": document.page_content}
        for document in documents
    ]

    lexical_validation = validate_answer(
        answer=answer,
        sources=validation_sources,
    )

    safety_result = {
        "approved": False,
        "requires_human_review": True,
        "reason": (
            "Verificações automáticas executadas. "
            "Elas não comprovam a precisão clínica. "
            "Aprovação humana obrigatória."
        ),
        "detected_issues": lexical_validation["detected_issues"],
        "lexical_check_passed": lexical_validation["lexical_check_passed"],
        "human_review_status": "pending",
    }

    return {
        "answer": answer,
        "sources": sources,
        "safety": safety_result,
    }
