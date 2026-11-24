from medical_assistant.safety import validate_answer


def test_resposta_sem_fontes_exige_revisao():
    result = validate_answer(
        answer="A leucemia afeta as células sanguíneas.",
        sources=[]
    )

    assert result["approved"] is False
    assert result["requires_human_review"] is True
    assert "no_retrieved_sources" in result["detected_issues"]


def test_afirmacao_sensivel_exige_revisao():
    result = validate_answer(
        answer="A cura é geralmente possível.",
        sources=[{"source": "MedQuAD"}]
    )

    assert result["approved"] is False
    assert result["requires_human_review"] is True


def test_expressao_inadequada_e_detectada():
    result = validate_answer(
        answer="A doença pode causar sangue roxo.",
        sources=[{"source": "MedQuAD"}]
    )

    assert result["approved"] is False
    assert result["requires_human_review"] is True
    assert any(
        "sangue roxo" in issue
        for issue in result["detected_issues"]
    )
