from medical_assistant.anonymization import anonymize_text, anonymize_data


def test_remove_email():
    result = anonymize_text("Contato: teste@example.com")
    assert "[EMAIL_REMOVIDO]" in result
    assert "teste@example.com" not in result


def test_remove_cpf():
    result = anonymize_text("CPF: 123.456.789-00")
    assert "[CPF_REMOVIDO]" in result
    assert "123.456.789-00" not in result


def test_remove_cnpj():
    result = anonymize_text("CNPJ: 12.345.678/0001-90")
    assert "[CNPJ_REMOVIDO]" in result
    assert "12.345.678/0001-90" not in result


def test_remove_cep():
    result = anonymize_text("CEP: 01310-100")
    assert "[CEP_REMOVIDO]" in result
    assert "01310-100" not in result


def test_remove_nome_informado():
    result = anonymize_text("Nome: Maria da Silva")
    assert "[NOME_REMOVIDO]" in result
    assert "Maria da Silva" not in result


def test_remove_data_nascimento():
    result = anonymize_text("Data de nascimento: 15/05/1990")
    assert "[DATA_REMOVIDA]" in result
    assert "15/05/1990" not in result


def test_anonimiza_dicionarios_e_listas_recursivamente():
    data = {
        "patient": {
            "email": "teste@example.com",
            "contacts": [
                "CPF: 123.456.789-00"
            ]
        },
        "age": 45
    }

    result = anonymize_data(data)

    assert "[EMAIL_REMOVIDO]" in result["patient"]["email"]
    assert "[CPF_REMOVIDO]" in result["patient"]["contacts"][0]
    assert result["age"] == 45


def test_preserva_valores_que_nao_sao_texto():
    assert anonymize_data(123) == 123
    assert anonymize_data(None) is None


def test_anonimiza_tuplas():
    result = anonymize_data(("teste@example.com", 45))

    assert isinstance(result, tuple)
    assert "[EMAIL_REMOVIDO]" in result[0]
    assert result[1] == 45
