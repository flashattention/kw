from typing import List, Union
from langchain.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.types import Command
from langgraph.graph import END
from pydantic import BaseModel, Field

from settings import get_model, State

MAX_REPLANS = 1  # 마무리 단계에서 계획을 보완할 수 있는 최대 횟수

class Plan(BaseModel):
    """아직 수행할 작업이 남아 있을 때 사용. 수행할 단계 목록을 담는다."""

    steps: List[str] = Field(
        description="따라야 할 단계들, 정렬된 순서여야 함"
    )

class ConversationalResponse(BaseModel):
    """더 수행할 작업이 없을 때만 사용. 사용자에게 줄 최종 답변을 담는다.
    앞으로 수행할 작업이나 계획을 여기에 쓰지 않는다."""

    response: str = Field(description="사용자의 질문에 대한 대화형 응답")

class PlanningResponse(BaseModel):
    final_output: Union[Plan, ConversationalResponse]

def planning_node(state: State) -> Command:
    """Planning 노드: 사용자 요청마다 새로 진입하여 계획 수립 (이전 요청의 plan/past_steps는 초기화)"""

    planner_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """사용자의 마지막 요청을 분석하여 단계별 작업 계획을 세워주세요.
            계획은 논리적 순서로 배열하고, 각 단계가 독립적으로 실행 가능하도록 구성하세요.
            각 단계는 'research' 에이전트(웹 검색, 정보 수집) 또는 'canvas' 에이전트(글 작성, 목차 작성, 파일 저장) 중 하나가 수행할 수 있어야 합니다.
            하나의 단계에 너무 많은 양의 작업이 몰리지 않도록 주의하세요.
            불필요한 단계는 추가하지 말고, 최종 단계의 결과가 사용자가 원하는 최종 답변이 되도록 하세요.
            단계의 개수는 2~5개 사이로 유지하세요.

            단, 별도의 조사나 글 작성 없이 바로 답할 수 있는 간단한 질문이나 대화라면 ConversationalResponse로 바로 답변하세요.
            """
        ),
        MessagesPlaceholder(variable_name="messages")
    ])

    planner = planner_prompt | get_model().with_structured_output(PlanningResponse)
    result = planner.invoke({"messages": state["messages"]})

    # 바로 답변할 수 있는 경우
    if isinstance(result.final_output, ConversationalResponse):
        return Command(
            goto=END,
            update={
                "plan": [],
                "past_steps": [],
                "messages": [AIMessage(content=result.final_output.response, name="planning")]
            }
        )

    generated_plan = result.final_output.steps
    plan_list = "\n".join([f"{i+1}. {task}" for i, task in enumerate(generated_plan)])

    return Command(
        goto="supervisor",
        update={
            "plan": generated_plan,
            "past_steps": [],
            "replans": 0,
            "messages": [AIMessage(content=f"작업 계획을 수립했습니다:\n{plan_list}", name="planning")]
        }
    )

def replanning_node(state: State) -> Command:
    """Replanning 노드: 모든 단계가 끝난 뒤 진입. 최종 답변을 만들거나, 목표 달성에 부족하면 계획을 보완 (최대 MAX_REPLANS회)"""

    messages = state["messages"]
    past_steps = state.get("past_steps", [])
    replans = state.get("replans", 0)
    can_replan = replans < MAX_REPLANS

    user_request = next((msg.content for msg in reversed(messages) if msg.type == "human"
                         and not msg.content.startswith("작업을 수행해주세요:")), "")
    past_steps_summary = "\n".join([f"- {task}: {result}" for task, result in past_steps]) or "완료된 작업이 없습니다."

    replan_guide = """
            **완료된 작업들로 사용자의 목표를 아직 달성하지 못했다면:**
            - Plan으로 추가로 수행할 단계만 제공하세요 (1~3개)
            - 이미 완료된 작업은 포함하지 마세요
            - 각 단계는 'canvas' 에이전트(글 작성, 콘텐츠 생성, 파일 저장) 또는 'research' 에이전트(웹 검색, 정보 수집)가 수행할 수 있어야 합니다
            """ if can_replan else ""

    replanner_prompt = ChatPromptTemplate.from_template(
        """계획한 작업이 모두 끝났습니다. 완료된 작업 결과를 바탕으로 판단하세요.

            목표:
            {input}

            완료한 작업들:
            {past_steps}

            **완료된 작업들로 사용자의 목표를 달성했다면:**
            - ConversationalResponse로 사용자에게 도움이 되는 최종 답변을 제공하세요
            - 작업 결과(조사 내용, 저장된 파일명 등)를 바탕으로 자연스럽게 정리하고, 앞으로 할 작업이나 계획은 쓰지 마세요
            {replan_guide}"""
    )

    schema = PlanningResponse if can_replan else ConversationalResponse
    result = (replanner_prompt | get_model().with_structured_output(schema)).invoke({
        "input": user_request,
        "past_steps": past_steps_summary,
        "replan_guide": replan_guide,
    })
    output = result.final_output if can_replan else result

    # 계획 보완: 추가 단계를 수행하러 supervisor로
    if isinstance(output, Plan) and output.steps:
        return Command(
            goto="supervisor",
            update={
                "plan": output.steps,
                "replans": replans + 1,
                "messages": [AIMessage(content=f"계획을 보완했습니다. 추가 작업:\n" + "\n".join(output.steps), name="planning")]
            }
        )

    # 최종 답변
    response = output.response if isinstance(output, ConversationalResponse) else "작업을 완료했습니다."
    return Command(
        goto=END,
        update={"messages": [AIMessage(content=response, name="planning")]}
    )
