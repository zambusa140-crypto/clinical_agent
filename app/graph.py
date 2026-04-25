import re
from datetime import datetime, timezone
from typing import Optional

from app.state import IntakeState
from app.schemas import HPI, ClinicalBrief


HPI_FIELDS = ["onset", "location", "duration", "character", "severity", "aggravating", "relieving"]

HPI_QUESTIONS = {
    "onset": "When did your symptoms first start?",
    "location": "Where exactly do you feel the pain or discomfort?",
    "duration": "How long does each episode last? Is it constant or intermittent?",
    "character": "Can you describe what the pain feels like?",
    "severity": "On a scale of 1 to 10, how severe is your pain?",
    "aggravating": "What makes your symptoms worse?",
    "relieving": "What helps relieve your symptoms?"
}

HPI_FIELD_CONTEXT = {
    "onset": "when your symptoms first started",
    "location": "where exactly you feel it",
    "duration": "how long each episode lasts",
    "character": "what the pain feels like",
    "severity": "how severe the pain is on a 1-10 scale",
    "aggravating": "what makes your symptoms worse",
    "relieving": "what helps relieve your symptoms",
}

CC_KEYWORDS_TO_ROS = {
    "chest": ["cardiac", "respiratory", "gi"],
    "pain": ["cardiac", "respiratory", "gi"],
    "headache": ["neuro", "ent", "vision"],
    "head": ["neuro", "ent", "vision"],
    "breath": ["respiratory", "cardiac"],
    "shortness": ["respiratory", "cardiac"],
    "cough": ["respiratory", "ent"],
    "dizzy": ["neuro", "cardiac"],
    "nausea": ["gi", "constitutional"],
    "vomiting": ["gi", "constitutional"],
}

DEFAULT_ROS = ["constitutional", "cardiac", "respiratory"]


def get_relevant_ros_systems(cc: str) -> list[str]:
    cc_lower = cc.lower()
    for keyword, systems in CC_KEYWORDS_TO_ROS.items():
        if keyword in cc_lower:
            return systems
    return DEFAULT_ROS


def extract_hpi_value(answer: str, field: str) -> str:
    answer = answer.strip()
    if field == "severity":
        match = re.search(r'(\d{1,2})\s*(?:out of|/)?\s*10', answer, re.IGNORECASE)
        if match:
            return f"{match.group(1)}/10"
    return answer


def _is_vague_answer(answer: str) -> bool:
    vague_phrases = ["i don't know", "not sure", "dont know", "idk", "maybe", "i guess"]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in vague_phrases)


class IntakeAgent:
    def __init__(self):
        pass

    def process_message(self, session_id: str, message: str, session_store: dict) -> tuple[str, str, Optional[ClinicalBrief]]:
        if session_id not in session_store:
            session_store[session_id] = {
                "messages": [],
                "chief_complaint": "",
                "hpi": {},
                "ros": {},
                "current_node": "intake",
                "clinical_brief": None,
                "ros_systems": [],
                "ros_current_index": 0,
                "ros_pending_system": None,
                "last_processed_message_index": 0,
                "vague_retry_field": None,
            }

        state = session_store[session_id]

        if message:
            state["messages"].append({"role": "user", "content": message})

        current_node = state.get("current_node", "intake")

        if current_node == "intake":
            state = self.intake_node(state)
            current_node = self.route_after_intake(state)
            if current_node != "intake":
                state["current_node"] = current_node

        elif current_node == "hpi":
            state = self.hpi_node(state)
            current_node = self.route_after_hpi(state)
            state["current_node"] = current_node

        elif current_node == "ros":
            state = self.ros_node(state)
            current_node = self.route_after_ros(state)
            state["current_node"] = current_node

        elif current_node == "brief_generator" or current_node == "done":
            state = self.brief_generator_node(state)

        session_store[session_id] = state

        reply = ""
        if state.get("messages"):
            for msg in reversed(state["messages"]):
                if msg["role"] == "assistant":
                    reply = msg["content"]
                    break

        current_node = state.get("current_node", "intake")
        brief = state.get("clinical_brief")

        return reply, current_node, brief

    def intake_node(self, state: IntakeState) -> IntakeState:
        messages = state.get("messages", [])
        last_idx = state.get("last_processed_message_index", 0)
        cc = state.get("chief_complaint", "")

        has_new_user_msg = (len(messages) > last_idx and messages[-1]["role"] == "user")

        if not cc and has_new_user_msg:
            state["chief_complaint"] = messages[-1]["content"]
            reply = f"I understand you're experiencing {state['chief_complaint']}. Let me ask you some questions about this."
            state["last_processed_message_index"] = len(messages)
        elif not cc:
            reply = "Hello, I'm conducting your pre-visit clinical intake. What brings you in today?"
        else:
            reply = "Moving to next section..."
            state["last_processed_message_index"] = len(messages)

        new_messages = messages + [{"role": "assistant", "content": reply}]
        state["messages"] = new_messages
        state["current_node"] = "intake"
        return state

    def hpi_node(self, state: IntakeState) -> IntakeState:
        messages = state.get("messages", [])
        last_idx = state.get("last_processed_message_index", 0)
        hpi = state.get("hpi", {})

        next_field = None
        for field in HPI_FIELDS:
            if field not in hpi or not hpi[field]:
                next_field = field
                break

        if next_field is None:
            reply = "Thank you for providing that information. Now let me ask about other symptoms."
            state["last_processed_message_index"] = len(messages)
            new_messages = messages + [{"role": "assistant", "content": reply}]
            state["messages"] = new_messages
            state["current_node"] = "hpi"
            return state

        has_new_user_msg = (len(messages) > last_idx and messages[-1]["role"] == "user")

        if has_new_user_msg:
            answer = messages[-1]["content"]

            if _is_vague_answer(answer):
                field_context = HPI_FIELD_CONTEXT.get(next_field, "your symptoms")
                reply = f"Could you be more specific about {field_context}?"
                state["vague_retry_field"] = next_field
                new_messages = messages + [{"role": "assistant", "content": reply}]
                state["messages"] = new_messages
                state["current_node"] = "hpi"
                return state

            hpi[next_field] = extract_hpi_value(answer, next_field)
            state["hpi"] = hpi
            state["vague_retry_field"] = None
            state["last_processed_message_index"] = len(messages)

            next_idx = HPI_FIELDS.index(next_field)
            if next_idx < len(HPI_FIELDS) - 1:
                next_q = HPI_FIELDS[next_idx + 1]
                reply = HPI_QUESTIONS[next_q]
            else:
                reply = "Thank you. Now let me ask about other associated symptoms."
        else:
            reply = HPI_QUESTIONS[next_field]

        new_messages = messages + [{"role": "assistant", "content": reply}]
        state["messages"] = new_messages
        state["current_node"] = "hpi"
        return state

    def ros_node(self, state: IntakeState) -> IntakeState:
        messages = state.get("messages", [])
        last_idx = state.get("last_processed_message_index", 0)
        ros = state.get("ros", {})
        cc = state.get("chief_complaint", "")

        if "ros_systems" not in state or not state.get("ros_systems"):
            state["ros_systems"] = get_relevant_ros_systems(cc)
            state["ros_current_index"] = 0
            state["ros_pending_system"] = None

        ros_systems = state["ros_systems"]
        current_idx = state.get("ros_current_index", 0)

        if current_idx >= len(ros_systems):
            reply = "Thank you. I have enough information to generate your clinical brief."
            state["last_processed_message_index"] = len(messages)
            new_messages = messages + [{"role": "assistant", "content": reply}]
            state["messages"] = new_messages
            state["current_node"] = "ros"
            return state

        has_new_user_msg = (len(messages) > last_idx and messages[-1]["role"] == "user")

        if has_new_user_msg:
            answer = messages[-1]["content"]
            pending_system = state.get("ros_pending_system")

            if pending_system:
                positive_findings = []
                negative_findings = []

                findings = [f.strip() for f in answer.split(",")]
                for f in findings:
                    if "no " in f.lower() or "none" in f.lower():
                        negative_findings.append(f)
                    else:
                        positive_findings.append(f)

                ros[pending_system] = positive_findings + negative_findings
                state["ros"] = ros
                state["ros_pending_system"] = None
                state["last_processed_message_index"] = len(messages)

            if current_idx < len(ros_systems):
                next_system = ros_systems[current_idx]
                reply = f"Let's review your {next_system} system. Any {next_system} symptoms? Please mention what's present and what's not."
                state["ros_current_index"] = current_idx + 1
                state["ros_pending_system"] = next_system
            else:
                reply = "Thank you. I have enough information."
        else:
            if current_idx < len(ros_systems):
                next_system = ros_systems[current_idx]
                reply = f"Let's start with your {next_system} system. Any {next_system} symptoms? Please mention what's present and what's not."
                state["ros_current_index"] = current_idx + 1
                state["ros_pending_system"] = next_system
            else:
                reply = "Continuing review of systems..."

        new_messages = messages + [{"role": "assistant", "content": reply}]
        state["messages"] = new_messages
        state["current_node"] = "ros"
        return state

    def brief_generator_node(self, state: IntakeState) -> IntakeState:
        ros = state.get("ros", {})

        hpi_data = state.get("hpi", {})
        hpi_obj = HPI(
            onset=hpi_data.get("onset", "not specified"),
            location=hpi_data.get("location", "not specified"),
            duration=hpi_data.get("duration", "not specified"),
            character=hpi_data.get("character", "not specified"),
            severity=hpi_data.get("severity", "not specified"),
            aggravating=hpi_data.get("aggravating", "not specified"),
            relieving=hpi_data.get("relieving", "not specified")
        )

        brief = ClinicalBrief(
            chief_complaint=state.get("chief_complaint", ""),
            hpi=hpi_obj,
            ros=ros,
            generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )

        state["clinical_brief"] = brief
        state["current_node"] = "done"

        reply = "Your clinical intake is complete. Here is your summary."
        state["messages"] = state.get("messages", []) + [{"role": "assistant", "content": reply}]

        return state

    def route_after_intake(self, state: IntakeState) -> str:
        cc = state.get("chief_complaint", "")
        if cc:
            return "hpi"
        return "intake"

    def route_after_hpi(self, state: IntakeState) -> str:
        hpi = state.get("hpi", {})
        for field in HPI_FIELDS:
            if field not in hpi or not hpi[field]:
                return "hpi"
        return "ros"

    def route_after_ros(self, state: IntakeState) -> str:
        ros_systems = state.get("ros_systems", [])
        current_idx = state.get("ros_current_index", 0)
        if current_idx >= len(ros_systems):
            return "brief_generator"
        return "ros"


agent = IntakeAgent()
