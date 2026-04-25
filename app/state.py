from typing import Optional, TypedDict

from app.schemas import ClinicalBrief


class IntakeState(TypedDict):
    messages: list[dict]
    chief_complaint: str
    hpi: dict
    ros: dict[str, list[str]]
    current_node: str
    clinical_brief: Optional[ClinicalBrief]
