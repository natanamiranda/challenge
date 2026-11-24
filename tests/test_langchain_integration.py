import json
from types import SimpleNamespace

import pytest

from medical_assistant import assistant


def make_document(content):
    return SimpleNamespace(
        page_content=content,
        metadata={
            "source": "MedQuAD",
            "source_url": "https://example.com/medical-source",
        },
    )


def disable_audit(monkeypatch):
    monkeypatch.setattr(
        assistant,
        "log_event",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        assistant,
        "log_error",
        lambda **kwargs: None,
    )


def test_generate_answer_rejeita_pergunta_vazia():
    with pytest.raises(ValueError, match="pergunta não pode estar vazia"):
        assistant.generate_answer("")


def test_rag_sem_documentos_exige_revisao_humana(monkeypatch):
    disable_audit(monkeypatch)

    monkeypatch.setattr(
        assistant,
        "search_protocols",
        lambda question, k=3: [],
    )

    result = assistant.generate_answer("Pergunta de teste")

    assert result["sources"] == []
    assert result["safety"]["approved"] is False
    assert result["safety"]["requires_human_review"] is True
    assert result["safety"]["human_review_status"] == "pending"
    assert "fontes suficientes" in result["answer"]


def test_pergunta_sobre_sintomas_usa_conteudo_recuperado(monkeypatch):
    disable_audit(monkeypatch)

    document = make_document(
        "Common symptoms include fever and night sweats."
    )

    monkeypatch.setattr(
        assistant,
        "search_protocols",
        lambda question, k=3: [document],
    )

    def fail_if_llm_called():
        pytest.fail("O LLM não deveria ser chamado neste teste.")

    monkeypatch.setattr(assistant, "create_llm", fail_if_llm_called)

    result = assistant.generate_answer(
        "Quais são os sintomas?"
    )

    assert "Febre" in result["answer"]
    assert "Suores noturnos" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["safety"]["requires_human_review"] is True
    assert result["safety"]["approved"] is False


def test_pergunta_sobre_definicao_usa_conteudo_recuperado(monkeypatch):
    disable_audit(monkeypatch)

    document = make_document(
        "These cells are also called leukemia cells."
    )

    monkeypatch.setattr(
        assistant,
        "search_protocols",
        lambda question, k=3: [document],
    )

    def fail_if_llm_called():
        pytest.fail("O LLM não deveria ser chamado neste teste.")

    monkeypatch.setattr(assistant, "create_llm", fail_if_llm_called)

    result = assistant.generate_answer(
        "O que é leucemia?"
    )

    assert "células leucêmicas" in result["answer"]
    assert result["safety"]["requires_human_review"] is True


def test_langchain_recebe_contexto_anonimizado(monkeypatch):
    disable_audit(monkeypatch)

    document = make_document(
        "The document contains general medical information."
    )

    captured = {}

    def fake_search(question, k=3):
        captured["search_question"] = question
        return [document]

    class FakeLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return SimpleNamespace(
                content="Resposta fictícia gerada pelo modelo."
            )

    monkeypatch.setattr(
        assistant,
        "search_protocols",
        fake_search,
    )

    monkeypatch.setattr(
        assistant,
        "load_medical_glossary",
        lambda: {},
    )

    monkeypatch.setattr(
        assistant,
        "create_llm",
        lambda: FakeLLM(),
    )

    question = (
        "Quais informações constam sobre o contato "
        "teste@example.com?"
    )

    patient_context = {
        "email": "paciente@example.com",
        "patient_code": "PAT-001",
    }

    result = assistant.generate_answer(
        question=question,
        patient_context=patient_context,
    )

    assert "teste@example.com" not in captured["search_question"]
    assert "teste@example.com" not in captured["prompt"]
    assert "paciente@example.com" not in captured["prompt"]

    assert result["answer"] == (
        "Resposta fictícia gerada pelo modelo."
    )

    assert result["safety"]["approved"] is False
    assert result["safety"]["requires_human_review"] is True
    assert result["safety"]["human_review_status"] == "pending"
    assert len(result["sources"]) == 1
