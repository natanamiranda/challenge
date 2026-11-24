import pytest

from medical_assistant import workflow


def fake_result(requires_review=True):
    return {
        "answer": "Resposta médica fictícia para teste.",
        "sources": [
            {
                "source": "MedQuAD",
                "url": "https://example.com/source"
            }
        ],
        "safety": {
            "approved": False,
            "requires_human_review": requires_review,
            "detected_issues": [],
        },
    }


def test_workflow_exige_revisao_humana(monkeypatch):
    monkeypatch.setattr(
        workflow,
        "generate_answer",
        lambda question, patient_context=None: fake_result(True),
    )

    result = workflow.run_medical_workflow(
        question="Pergunta de teste",
        patient_context={"patient_code": "PAT-001"},
    )

    assert result["workflow_status"] == "awaiting_human_review"
    assert result["safety"]["requires_human_review"] is True
    assert result["safety"]["approved"] is False
    assert result["safety"]["human_review_status"] == "pending"
    assert result["answer"] == "Resposta médica fictícia para teste."
    assert len(result["sources"]) == 1


def test_workflow_encaminha_resposta_sem_revisao(monkeypatch):
    monkeypatch.setattr(
        workflow,
        "generate_answer",
        lambda question, patient_context=None: fake_result(False),
    )

    result = workflow.run_medical_workflow(
        question="Pergunta de teste",
    )

    assert result["workflow_status"] == "completed"
    assert result["safety"]["requires_human_review"] is False


def test_workflow_rejeita_pergunta_vazia():
    with pytest.raises(ValueError, match="A pergunta não pode estar vazia"):
        workflow.run_medical_workflow("")


def test_workflow_rejeita_pergunta_com_espacos():
    with pytest.raises(ValueError, match="A pergunta não pode estar vazia"):
        workflow.run_medical_workflow("   ")


def test_workflow_envia_contexto_para_geracao(monkeypatch):
    captured = {}

    def fake_generate(question, patient_context=None):
        captured["question"] = question
        captured["patient_context"] = patient_context
        return fake_result(True)

    monkeypatch.setattr(
        workflow,
        "generate_answer",
        fake_generate,
    )

    context = {"patient_code": "PAT-001"}

    workflow.run_medical_workflow(
        question="Pergunta de teste",
        patient_context=context,
    )

    assert captured["question"] == "Pergunta de teste"
    assert captured["patient_context"] == context
