
import re


PATTERNS = [
    # E-mail
    (
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        '[EMAIL_REMOVIDO]'
    ),

    # CPF
    (
        r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b',
        '[CPF_REMOVIDO]'
    ),

    # CNPJ
    (
        r'\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b',
        '[CNPJ_REMOVIDO]'
    ),

    # Telefone brasileiro
    (
        r'(?<!\d)(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?9?\d{4}[- ]?\d{4}(?!\d)',
        '[TELEFONE_REMOVIDO]'
    ),

    # CEP
    (
        r'\b\d{5}-?\d{3}\b',
        '[CEP_REMOVIDO]'
    ),
]


def anonymize_text(value):
    """
    Remove ou substitui identificadores pessoais em textos.
    """

    if not isinstance(value, str):
        return value

    result = value

    for pattern, replacement in PATTERNS:
        result = re.sub(pattern, replacement, result)

    # Nome informado em campos textuais
    result = re.sub(
        r'(?i)(nome\s*[:=]\s*)[^,\n;]+',
        r'\1[NOME_REMOVIDO]',
        result
    )

    # Data de nascimento
    result = re.sub(
        r'(?i)(data\s+de\s+nascimento\s*[:=]\s*)\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        r'\1[DATA_REMOVIDA]',
        result
    )

    return result


def anonymize_data(data):
    """
    Anonimiza recursivamente strings em dicionários e listas.
    """

    if isinstance(data, str):
        return anonymize_text(data)

    if isinstance(data, dict):
        return {
            key: anonymize_data(value)
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [
            anonymize_data(item)
            for item in data
        ]

    if isinstance(data, tuple):
        return tuple(
            anonymize_data(item)
            for item in data
        )

    return data
