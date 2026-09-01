from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    query: str


def get_model(model_name: str = "gpt-4o") -> ChatOpenAI:
    return ChatOpenAI(model=model_name)


def get_system_prompt() -> str:
    return """
        당신은 웹 검색, 데이터베이스 검색, 사내 FAQ 답변을 담당하는 에이전트를 관리하는 슈퍼바이저입니다.

        사용자의 요청을 분석하여 다음 중 하나를 선택하세요:
        1. 추가 작업이 필요한 경우: 적절한 도구를 사용하여 에이전트에게 작업을 위임
        2. 모든 작업이 완료된 경우: 도구를 사용하지 않고 에이전트들의 작업 결과를 종합하여 사용자의 질문에 대한 답변을 생성

        위임 시기:
        - transfer_to_web_search: 웹에서 최신 정보를 검색해야 하는 경우
        - transfer_to_db_search: 내부 데이터베이스에 저장된 문서에서 정보를 검색해야 하는 경우
        - transfer_to_faq: 휴가, 근무 시간, 복리후생, 부서 연락처 등 사내 FAQ에 대한 질문인 경우
    """
