from typing import Literal, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from settings import get_model, State

class Router(TypedDict):
    """작업을 수행할 에이전트를 지정합니다."""
    next: Literal["canvas", "research"]

def supervisor_node(state: State) -> Command[Literal["canvas", "research", "replanning"]]:
    """Supervisor 노드: 현재 작업(plan[0])을 수행할 에이전트만 선택. 단계 완료 처리는 에이전트 노드가 코드로 수행"""

    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])

    # 남은 작업이 없으면 마무리 단계로
    if not plan:
        return Command(goto="replanning")

    task_description = plan[0]
    done_tasks = "\n".join(f"- {task}" for task, _ in past_steps) or "없음"

    system_prompt = f"""
        당신은 멀티 에이전트 시스템의 슈퍼바이저입니다.
        Router 도구를 사용하여 현재 작업을 수행할 에이전트를 지정하세요.

        ## 현재 작업:
        {task_description}

        ## 이미 완료된 작업:
        {done_tasks}

        ## 팀 멤버:
        - canvas: 글 작성, 목차 작성, 파일 저장을 담당
        - research: 웹 검색 기반 분석, 자료 조사 등을 담당
        """

    llm = get_model("supervisor", reasoning_effort="none")
    response = llm.bind_tools([Router], tool_choice="Router").invoke([{"role": "system", "content": system_prompt}])
    goto = response.tool_calls[0]["args"]["next"]

    return Command(
        goto=goto,
        update={"messages": [HumanMessage(content=f"작업을 수행해주세요: {task_description}")]}
    )
