
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_assistant.database import SessionLocal
from medical_assistant.patient_service import get_patient_context


def main():
    db = SessionLocal()

    try:
        patient = get_patient_context(db, "PAT-001")

        if patient is None:
            print("Paciente não encontrado.")
            return

        print("CONSULTA DO PACIENTE")
        print(json.dumps(patient, indent=4, ensure_ascii=False))

    finally:
        db.close()


if __name__ == "__main__":
    main()
