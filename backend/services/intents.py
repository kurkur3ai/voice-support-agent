import json
import os

HUMAN_HANDOFF = "human_handoff"

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "intents.json")

with open(_DATA_PATH, "r", encoding="utf-8") as _f:
    INTENTS: dict[str, dict] = json.load(_f)


def get_response(intent: str) -> str:
    return INTENTS.get(intent, INTENTS[HUMAN_HANDOFF])["response"]
