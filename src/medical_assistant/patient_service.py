
from sqlalchemy.orm import Session, joinedload

from medical_assistant.models import Patient, Exam


def get_patient_by_code(db: Session, patient_code: str):
    return (
        db.query(Patient)
        .options(
            joinedload(Patient.exams),
            joinedload(Patient.allergies),
            joinedload(Patient.conditions),
        )
        .filter(Patient.patient_code == patient_code)
        .first()
    )


def get_pending_exams(db: Session, patient_id: int):
    return (
        db.query(Exam)
        .filter(
            Exam.patient_id == patient_id,
            Exam.status == "pendente",
        )
        .order_by(Exam.requested_date)
        .all()
    )


def get_patient_context(db: Session, patient_code: str):
    patient = get_patient_by_code(db, patient_code)

    if not patient:
        return None

    pending_exams = get_pending_exams(db, patient.id)

    return {
        "patient_code": patient.patient_code,
        "name": patient.name,
        "age": patient.age,
        "sex": patient.sex,
        "clinical_summary": patient.clinical_summary,
        "allergies": [
            {
                "substance": allergy.substance,
                "reaction": allergy.reaction,
            }
            for allergy in patient.allergies
        ],
        "conditions": [
            condition.description
            for condition in patient.conditions
            if condition.active
        ],
        "pending_exams": [
            {
                "exam_name": exam.exam_name,
                "requested_date": exam.requested_date.isoformat(),
            }
            for exam in pending_exams
        ],
    }
