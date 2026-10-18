from typing import List
from langgraph.graph import MessagesState

class State(MessagesState):
    plan: List[str]            # 남은 단계 (plan[0]이 현재 작업)
    past_steps: List[tuple]    # 완료된 (단계, 결과) 목록
    replans: int               # 마무리 단계에서 계획을 보완한 횟수

from langchain_openai import ChatOpenAI

MODELS = {"supervisor": "gpt-5.4-nano", "default": "gpt-5.4-mini"}

def get_model(role: str = "default", **kwargs) -> ChatOpenAI:
    return ChatOpenAI(model=MODELS.get(role, MODELS["default"]), **kwargs)

def complete_step(state: State, result: str) -> dict:
    """에이전트가 현재 작업(plan[0])을 수행한 뒤 호출: 해당 단계를 plan에서 빼고 past_steps에 기록"""

    plan = state.get("plan", [])
    if not plan:
        return {}

    return {
        "plan": plan[1:],
        "past_steps": state.get("past_steps", []) + [(plan[0], result)],
    }
