
from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END

from medical_assistant.assistant import generate_answer


class MedicalWorkflowState(TypedDict, total=False):
    question: str
    patient_context: dict[str, Any] | None
    result: dict[str, Any]
    workflow_status: str
    error: str


def prepare_consultation(state: MedicalWorkflowState):
    question = state.get("question", "")

    if not question or not question.strip():
        raise ValueError("A pergunta não pode estar vazia.")

    return {
        "workflow_status": "consultation_prepared"
    }


def generate_response(state: MedicalWorkflowState):
    result = generate_answer(
        question=state["question"],
        patient_context=state.get("patient_context"),
    )

    return {
        "result": result,
        "workflow_status": "response_generated",
    }


def route_after_generation(state: MedicalWorkflowState):
    result = state.get("result", {})
    safety = result.get("safety", {})

    if safety.get("requires_human_review", True):
        return "human_review"

    return "completed"


def require_human_review(state: MedicalWorkflowState):
    result = state.get("result", {})
    safety = result.get("safety", {})

    safety["approved"] = False
    safety["requires_human_review"] = True
    safety["human_review_status"] = "pending"

    result["safety"] = safety

    return {
        "result": result,
        "workflow_status": "awaiting_human_review",
    }


def complete_workflow(state: MedicalWorkflowState):
    return {
        "workflow_status": "completed"
    }


def build_medical_workflow():
    graph = StateGraph(MedicalWorkflowState)

    graph.add_node(
        "prepare_consultation",
        prepare_consultation,
    )

    graph.add_node(
        "generate_response",
        generate_response,
    )

    graph.add_node(
        "human_review",
        require_human_review,
    )

    graph.add_node(
        "completed",
        complete_workflow,
    )

    graph.add_edge(
        START,
        "prepare_consultation",
    )

    graph.add_edge(
        "prepare_consultation",
        "generate_response",
    )

    graph.add_conditional_edges(
        "generate_response",
        route_after_generation,
        {
            "human_review": "human_review",
            "completed": "completed",
        },
    )

    graph.add_edge(
        "human_review",
        END,
    )

    graph.add_edge(
        "completed",
        END,
    )

    return graph.compile()


medical_workflow = build_medical_workflow()


def run_medical_workflow(
    question: str,
    patient_context: dict | None = None,
):
    initial_state: MedicalWorkflowState = {
        "question": question,
        "patient_context": patient_context,
    }

    final_state = medical_workflow.invoke(initial_state)

    return {
        "answer": final_state["result"]["answer"],
        "sources": final_state["result"]["sources"],
        "safety": final_state["result"]["safety"],
        "workflow_status": final_state["workflow_status"],
    }
