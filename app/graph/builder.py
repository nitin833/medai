"""LangGraph builder — wires nodes and conditional edges."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.hitl.interrupts import image_hitl_node, info_hitl_node, safety_hitl_node
from app.graph.nodes.image_classifier import image_classifier_node
from app.graph.nodes.image_tool import image_tool_node
from app.graph.nodes.info_tool import info_tool_node
from app.graph.nodes.main_agent import main_agent_node
from app.graph.nodes.prescription import prescription_node
from app.graph.nodes.rag_tool import rag_tool_node
from app.graph.nodes.response import generate_response_node
from app.graph.nodes.safety import safety_check_node
from app.graph.state import MedAIState


def route_by_intent(state: MedAIState) -> str:
    return state.get("intent") or "rag"


def route_after_image_hitl(state: MedAIState) -> str:
    if state.get("hitl_action") == "reject":
        return "safety_check"
    return "image_classifier"


def needs_prescription(state: MedAIState) -> str:
    if state.get("detected_disease"):
        return "prescription"
    return "safety_check"


def route_by_risk(state: MedAIState) -> str:
    if state.get("risk_level") == "high":
        return "safety_hitl"
    return "generate_response"


def build_graph(*, checkpointer=None):
    builder = StateGraph(MedAIState)

    builder.add_node("main_agent", main_agent_node)
    builder.add_node("image_tool", image_tool_node)
    builder.add_node("image_hitl", image_hitl_node)
    builder.add_node("image_classifier", image_classifier_node)
    builder.add_node("info_tool", info_tool_node)
    builder.add_node("info_hitl", info_hitl_node)
    builder.add_node("rag_tool", rag_tool_node)
    builder.add_node("prescription", prescription_node)
    builder.add_node("safety_check", safety_check_node)
    builder.add_node("safety_hitl", safety_hitl_node)
    builder.add_node("generate_response", generate_response_node)

    builder.add_edge(START, "main_agent")
    builder.add_conditional_edges(
        "main_agent",
        route_by_intent,
        {
            "image": "image_tool",
            "info": "info_tool",
            "rag": "rag_tool",
            "direct": "safety_check",
        },
    )

    builder.add_edge("image_tool", "image_hitl")
    builder.add_conditional_edges(
        "image_hitl",
        route_after_image_hitl,
        {
            "image_classifier": "image_classifier",
            "safety_check": "safety_check",
        },
    )
    builder.add_edge("image_classifier", "rag_tool")

    builder.add_edge("info_tool", "info_hitl")
    builder.add_edge("info_hitl", "safety_check")

    builder.add_conditional_edges(
        "rag_tool",
        needs_prescription,
        {
            "prescription": "prescription",
            "safety_check": "safety_check",
        },
    )
    builder.add_edge("prescription", "safety_check")

    builder.add_conditional_edges(
        "safety_check",
        route_by_risk,
        {
            "safety_hitl": "safety_hitl",
            "generate_response": "generate_response",
        },
    )
    builder.add_edge("safety_hitl", "generate_response")
    builder.add_edge("generate_response", END)

    saver = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=saver)
