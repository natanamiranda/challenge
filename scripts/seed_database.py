
import sys
from pathlib import Path
from datetime import date, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_assistant.database import Base, engine, SessionLocal
from medical_assistant.models import Patient, Exam, Allergy, Condition


def seed_database():
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()

    try:
        if db.query(Patient).first():
            print("Banco já possui pacientes. Nenhum dado foi duplicado.")
            return

        patient1 = Patient(
            patient_code="PAT-001",
            name="Paciente Fictício 001",
            age=45,
            sex="Feminino",
            clinical_summary="Paciente fictícia em acompanhamento clínico."
        )

        patient1.exams = [
            Exam(
                exam_name="Hemograma completo",
                status="pendente",
                requested_date=date.today() - timedelta(days=2),
            ),
            Exam(
                exam_name="Glicemia em jejum",
                status="concluido",
                requested_date=date.today() - timedelta(days=5),
                result_summary="Resultado fictício: 98 mg/dL."
            ),
        ]

        patient1.allergies = [
            Allergy(
                substance="Penicilina",
                reaction="Reação alérgica fictícia registrada."
            )
        ]

        patient1.conditions = [
            Condition(description="Hipertensão arterial - dado fictício")
        ]

        patient2 = Patient(
            patient_code="PAT-002",
            name="Paciente Fictício 002",
            age=62,
            sex="Masculino",
            clinical_summary="Paciente fictício em acompanhamento ambulatorial."
        )

        patient2.exams = [
            Exam(
                exam_name="Eletrocardiograma",
                status="pendente",
                requested_date=date.today() - timedelta(days=1),
            )
        ]

        patient2.allergies = []

        patient2.conditions = [
            Condition(description="Diabetes tipo 2 - dado fictício")
        ]

        patient3 = Patient(
            patient_code="PAT-003",
            name="Paciente Fictício 003",
            age=34,
            sex="Feminino",
            clinical_summary="Paciente fictícia em avaliação de rotina."
        )

        patient3.exams = [
            Exam(
                exam_name="Ultrassonografia abdominal",
                status="concluido",
                requested_date=date.today() - timedelta(days=3),
                result_summary="Laudo fictício: sem alterações relevantes."
            )
        ]

        db.add_all([patient1, patient2, patient3])
        db.commit()

        print("Banco de dados criado com sucesso!")
        print("3 pacientes fictícios cadastrados.")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
