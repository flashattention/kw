from typing import List
from langgraph.graph import MessagesState

class State(MessagesState):
    plan: List[str]
    past_stepts: List[tuple]

class SupervisorState(MessagesState):
    plan: List[str]
    next: str
    past_steps: List[tuple]