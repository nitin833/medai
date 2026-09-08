from typing import Annotated, Literal, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

Intent = Literal["image", "info", "rag", "direct"]
RiskLevel = Literal["low", "medium", "high"]


class MedAIState(TypedDict, total=False):
    messages: Annotated[list, add_messages]

    intent: Optional[Intent]

    image_path: Optional[str]
    image_type: Optional[str]
    detected_disease: Optional[str]
    confidence: Optional[float]
    model_used: Optional[str]
    classification_details: Optional[list]

    patient_info: Optional[dict]
    retrieved_docs: Optional[list]
    rag_context: Optional[str]

    candidate_drugs: Optional[list]
    prescription_draft: Optional[str]

    risk_level: Optional[RiskLevel]
    safety_flags: Optional[list]

    hitl_pending: Optional[bool]
    hitl_action: Optional[str]
    human_feedback: Optional[str]

    final_response: Optional[str]
