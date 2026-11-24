
from sqlalchemy import (
    Column,
    Integer,
    String,
    Date,
    Text,
    ForeignKey,
    Boolean,
)
from sqlalchemy.orm import relationship

from medical_assistant.database import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_code = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    sex = Column(String(20))
    clinical_summary = Column(Text)

    exams = relationship(
        "Exam",
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    allergies = relationship(
        "Allergy",
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    conditions = relationship(
        "Condition",
        back_populates="patient",
        cascade="all, delete-orphan",
    )


class Exam(Base):
    __tablename__ = "exams"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    exam_name = Column(String(150), nullable=False)
    status = Column(String(30), nullable=False)
    requested_date = Column(Date, nullable=False)
    result_summary = Column(Text, nullable=True)

    patient = relationship(
        "Patient",
        back_populates="exams",
    )


class Allergy(Base):
    __tablename__ = "allergies"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    substance = Column(String(100), nullable=False)
    reaction = Column(Text)

    patient = relationship(
        "Patient",
        back_populates="allergies",
    )


class Condition(Base):
    __tablename__ = "conditions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    description = Column(Text, nullable=False)
    active = Column(Boolean, default=True, nullable=False)

    patient = relationship(
        "Patient",
        back_populates="conditions",
    )
