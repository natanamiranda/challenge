
import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_assistant.workflow import run_medical_workflow
from medical_assistant.database import SessionLocal
from medical_assistant.patient_service import get_patient_context


st.set_page_config(
    page_title="Assistente Médico com IA",
    page_icon="🏥",
    layout="wide",
)


st.title("🏥 Assistente Médico com IA")

st.caption(
    "Assistente de informação médica com busca em fontes científicas, "
    "contexto de pacientes fictícios e revisão humana obrigatória."
)

st.warning(
    "Protótipo acadêmico. Todos os pacientes são fictícios. "
    "As respostas são rascunhos informativos e não substituem "
    "a avaliação de um profissional de saúde."
)


# ============================================================
# SELEÇÃO E CONTEXTO DO PACIENTE
# ============================================================

st.sidebar.header("Paciente")

patient_code = st.sidebar.selectbox(
    "Selecione um paciente fictício",
    ["PAT-001", "PAT-002", "PAT-003"],
)

db = SessionLocal()

try:
    patient_context = get_patient_context(
        db,
        patient_code,
    )
finally:
    db.close()


if patient_context:
    st.subheader("Dados do paciente")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Código do paciente",
        patient_context["patient_code"],
    )

    col2.metric(
        "Idade",
        f'{patient_context["age"]} anos',
    )

    col3.metric(
        "Sexo",
        patient_context["sex"],
    )

    st.write(
        "**Nome:**",
        patient_context["name"],
    )

    st.write(
        "**Resumo clínico:**",
        patient_context["clinical_summary"],
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Alergias")

        if patient_context["allergies"]:
            for allergy in patient_context["allergies"]:
                st.write(
                    f'**{allergy["substance"]}:** '
                    f'{allergy["reaction"]}'
                )
        else:
            st.write("Nenhuma alergia registrada.")

    with col2:
        st.markdown("### Condições clínicas")

        if patient_context["conditions"]:
            for condition in patient_context["conditions"]:
                st.write(f"- {condition}")
        else:
            st.write("Nenhuma condição registrada.")

    st.markdown("### Exames pendentes")

    if patient_context["pending_exams"]:
        exams = [
            {
                "Exame": exam["exam_name"],
                "Data da solicitação": exam["requested_date"],
                "Situação": "Pendente",
            }
            for exam in patient_context["pending_exams"]
        ]

        st.dataframe(
            exams,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Não há exames pendentes.")

else:
    st.error("Paciente não encontrado no banco de dados.")


# ============================================================
# CONSULTA AO ASSISTENTE
# ============================================================

st.divider()

st.subheader("Consulta ao assistente")

question = st.text_area(
    "Digite sua pergunta",
    placeholder=(
        "Ex.: Quais são os sintomas da leucemia "
        "linfoblástica aguda em adultos?"
    ),
    height=110,
)


if st.button("Consultar", type="primary"):

    if not question.strip():
        st.warning("Digite uma pergunta antes de consultar.")

    elif not patient_context:
        st.error(
            "Não foi possível carregar o contexto do paciente."
        )

    else:
        with st.spinner(
            "Executando workflow LangGraph, consultando fontes "
            "e gerando resposta..."
        ):
            try:
                result = run_medical_workflow(
                    question=question,
                    patient_context=patient_context,
                )

                # ------------------------------------------------
                # RESPOSTA
                # ------------------------------------------------

                st.markdown("### Resposta gerada")

                st.write(result["answer"])

                # ------------------------------------------------
                # FONTES
                # ------------------------------------------------

                st.markdown("### Fontes consultadas")

                if result["sources"]:

                    for index, source in enumerate(
                        result["sources"],
                        start=1,
                    ):
                        source_name = (
                            source.get("source")
                            or "Não identificada"
                        )

                        with st.expander(
                            f"Fonte {index}: {source_name}",
                            expanded=True,
                        ):

                            if source.get("url"):
                                st.markdown(
                                    "[Acessar documento original]"
                                    f"({source['url']})"
                                )

                else:
                    st.info(
                        "Nenhuma fonte médica foi encontrada."
                    )

                # ------------------------------------------------
                # WORKFLOW LANGGRAPH
                # ------------------------------------------------

                st.markdown("### Workflow LangGraph")

                workflow_status = result.get(
                    "workflow_status",
                    "not_available",
                )

                workflow_labels = {
                    "awaiting_human_review": (
                        "Aguardando revisão humana"
                    ),
                    "completed": "Fluxo concluído",
                }

                st.info(
                    workflow_labels.get(
                        workflow_status,
                        workflow_status,
                    )
                )

                st.caption(
                    "O workflow executa preparação da consulta, "
                    "geração da resposta e encaminhamento condicional "
                    "para revisão humana."
                )

                # ------------------------------------------------
                # SEGURANÇA
                # ------------------------------------------------

                st.markdown("### Verificação de segurança")

                safety = result["safety"]

                st.error(
                    "Esta resposta aguarda revisão de um "
                    "profissional de saúde."
                )

                approval = (
                    "Aprovada"
                    if safety["approved"]
                    else "Não aprovada"
                )

                review_status = {
                    "pending": "Aguardando revisão humana",
                    "approved": "Aprovada por revisor humano",
                    "rejected": "Rejeitada por revisor humano",
                }.get(
                    safety["human_review_status"],
                    "Status não identificado",
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Aprovação automática",
                    approval,
                )

                col2.metric(
                    "Revisão humana",
                    review_status,
                )

                st.write(
                    "**Justificativa:**",
                    safety["reason"],
                )

                issues = safety.get(
                    "detected_issues",
                    [],
                )

                if issues:
                    st.markdown("**Alertas identificados:**")

                    for issue in issues:
                        st.write(f"- {issue}")

                else:
                    st.success(
                        "Nenhum problema identificado "
                        "pela validação lexical."
                    )

                st.info(
                    "A verificação automática é limitada e não "
                    "garante a precisão clínica ou a segurança "
                    "da resposta."
                )

            except Exception as error:
                st.error(
                    f"Não foi possível consultar o assistente: {error}"
                )


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    "Assistente Médico com IA | Protótipo acadêmico | "
    "Base de conhecimento MedQuAD | Dados sintéticos | "
    "LangChain | LangGraph"
)
