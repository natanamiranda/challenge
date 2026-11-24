import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"
AUDIT_FILE = AUDIT_DIR / "medical_assistant_audit.jsonl"


def get_audit_logger():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("medical_assistant.audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.FileHandler(
            AUDIT_FILE,
            encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger


def hash_value(value):
    if value is None or value == "":
        return None

    return hashlib.sha256(
        str(value).encode("utf-8")
    ).hexdigest()


def log_event(
    event,
    patient_code=None,
    question=None,
    sources=None,
    safety=None,
    status="success",
    details=None
):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "status": status,
        "patient_reference": hash_value(patient_code),
        "question_hash": hash_value(question),
        "sources": [
            {
                "source": item.get("source"),
                "url": item.get("url")
            }
            for item in (sources or [])
        ],
        "safety": safety or {},
        "details": details or {}
    }

    logger = get_audit_logger()

    logger.info(
        json.dumps(record, ensure_ascii=False, default=str)
    )


def log_error(
    event,
    error,
    patient_code=None,
    question=None
):
    log_event(
        event=event,
        patient_code=patient_code,
        question=question,
        status="error",
        details={
            "error_type": type(error).__name__,
            "error_message": "Detalhes omitidos por segurança."
        }
    )
