from typing import Optional, TypedDict

from app.schemas import ClinicalBrief


class IntakeState(TypedDict):
    messages: list[dict]
    chief_complaint: str
    hpi: dict
    ros: dict[str, list[str]]
    current_node: str
    clinical_brief: Optional[ClinicalBrief]
    ros_systems: list[str]
    ros_current_index: int
    ros_pending_system: Optional[str]
    last_processed_message_index: int
