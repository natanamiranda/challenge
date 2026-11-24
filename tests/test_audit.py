import json

from medical_assistant import audit


def test_hash_value():
    result = audit.hash_value("PAT-001")

    assert result is not None
    assert len(result) == 64
    assert result == audit.hash_value("PAT-001")


def test_hash_value_vazio():
    assert audit.hash_value(None) is None
    assert audit.hash_value("") is None


def test_log_event_grava_jsonl(monkeypatch, tmp_path):
    audit_file = tmp_path / "audit.jsonl"

    monkeypatch.setattr(audit, "AUDIT_DIR", tmp_path)
    monkeypatch.setattr(audit, "AUDIT_FILE", audit_file)

    logger = audit.logging.getLogger("medical_assistant.audit")

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    try:
        audit.log_event(
            event="consultation_test",
            patient_code="PAT-001",
            question="Pergunta de teste",
            sources=[{
                "source": "MedQuAD",
                "url": "https://example.com"
            }],
            safety={"requires_human_review": True},
            details={"test": True}
        )

        assert audit_file.exists()

        record = json.loads(
            audit_file.read_text(encoding="utf-8").strip()
        )

        assert record["event"] == "consultation_test"
        assert record["status"] == "success"
        assert record["patient_reference"] == audit.hash_value("PAT-001")
        assert record["question_hash"] == audit.hash_value("Pergunta de teste")
        assert record["sources"][0]["source"] == "MedQuAD"
        assert record["safety"]["requires_human_review"] is True
        assert record["details"]["test"] is True

        assert "PAT-001" not in json.dumps(record)
        assert "Pergunta de teste" not in json.dumps(record)

    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()


def test_log_error_registra_status_de_erro(monkeypatch, tmp_path):
    audit_file = tmp_path / "audit.jsonl"

    monkeypatch.setattr(audit, "AUDIT_DIR", tmp_path)
    monkeypatch.setattr(audit, "AUDIT_FILE", audit_file)

    logger = audit.logging.getLogger("medical_assistant.audit")

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    try:
        audit.log_error(
            event="consultation_error",
            error=ValueError("Erro de teste"),
            patient_code="PAT-001"
        )

        record = json.loads(
            audit_file.read_text(encoding="utf-8").strip()
        )

        assert record["event"] == "consultation_error"
        assert record["status"] == "error"
        assert record["details"]["error_type"] == "ValueError"
        assert record["details"]["error_message"] == "Detalhes omitidos por segurança."

    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
