import os

from typing import Annotated, Literal, Dict
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.types import Command
from langchain_core.messages import SystemMessage
from settings import get_model
from datetime import datetime

@tool
def create_outline_from_research(
    topic: Annotated[str, "목차를 작성할 주제"],
    research_results: Annotated[str, "research_agent가 조사한 내용들 (이전 메시지 히스토리에서 추출)"]
) -> str:
    """research_agent가 조사한 내용을 바탕으로 체계적인 목차를 작성합니다.
    
    조사 결과물의 내용을 분석하여 논리적인 흐름으로 재구성합니다.
    """

    llm = get_model()

    prompt = f"""
    주제: {topic}

    Research Agent가 조사한 내용:
    {research_results}

    위 조사 결과를 바탕으로 체계적이고 논리적인 목차를 작성해주세요.

    요구 사항:
    1. 조사된 내용을 모두 활용하여 목차 구성
    2. 독자가 이해하기 쉬운 순서로 재배치
    3. 각 챕터가 조사 결과의 어떤 부분을 다룰지 명확하게 표시
    4. 번호와 소제목으로 명확하게 구분 (1, 1.1, 1.2 등)
    5. 너무 세분화하지 말고 적절한 수준으로 그룹핑

    목차 형식 예시:
    1. [조사 결과 기반 주요 내용]
        1.1 [조사된 세부 주제 1]
    2. [조사 결과 기반 주요 내용]
        2.1 [조사된 세부 주제 1]
        2.2 [조사된 세부 주제 2]
    """

    response = llm.invoke(prompt)
    return f"조사 결과 기반 목차 작성 완료:\n\n{response.content}"

@tool
def write_content_from_research(
    outline: Annotated[str, "작성된 목차"],
    research_results: Annotated[str, "research_agent가 조사한 모든 내용"],
    topic: Annotated[str, "글의 주제"]
) -> str:
    """목차에 맞춰 조사 결과를 재구성하여 완성된 글을 작성합니다.
    
    조사 결과의 출처 링크를 반드시 포함하며, 목차 구조에 맞게 내용을 정리합니다.
    """

    llm = get_model()

    prompt = f"""
    주제: {topic}

    목차:
    {outline}

    Research Agent가 조사한 내용:
    {research_results}

    위 목차에 맞춰 조사된 내용을 재구성하여 **Markdown 형식**으로 완성된 글을 작성해주세요.

    작성 지침:
    1. 목차의 각 섹션에 맞게 조사 결과를 배치
    2. 조사된 정보의 출처 URL을 반드시 포함 (Markdown 링크 형식: [텍스트](URL) 또는 각 섹션 끝에 "참고: [URL]")
    3. 자연스러운 문장으로 재구성 (단순 복사 붙여넣기 금지)
    4. 목차 번호를 유지하면서 각 섹션을 작성
    5. 서론과 결론도 조사 내용을 바탕으로 작성
    6. 전체적으로 일관성 있는 흐름 유지
    7. Markdown 문법 활용
    - 제목: # (h1), ## (h2), ### (h3)
    - 강조: **굵게**, *기울임*
    - 링크: [텍스트](URL)
    - 목록: -, *, 1.
    - 인용: > 텍스트
    - 코드: `인라인 코드`, ```언어 블록 코드```
    - 구분선: ---
    - 출처는 Markdown 링크 형식으로: [출처명](URL)

    Markdown 출력 형식:
    ---
    # {topic}

    ## 1. 서론
    [내용...]

    ## 2. [메인 주제]

    ### 2.1 [세부 주제]
    [조사 결과 기반 내용...]

    **참고 자료:**
    - [출처명](URL)
    - [출처명2](URL2)

    ---

    ## 참고 자료
    - [모든 출처 URL을 Markdown 링크 형식으로 정리]
    """

    response = llm.invoke(prompt)
    return response.content

@tool
def save_to_file(
    filename: Annotated[str, "저장할 파일명 (.md)"],
    content: Annotated[str, """
    저장할 Markdown 형식의 내용 (목차, 링크, 출처 포함)
    - 제목: # (h1), ## (h2), ### (h3)
    - 강조: **굵게**, *기울임*
    - 링크: [텍스트](URL)
    - 목록: -, *, 1.
    - 인용: > 텍스트
    - 코드: `인라인 코드`, ```언어 블록 코드```
    - 구분선: ---
    - 출처는 Markdown 링크 형식으로: [출처명](URL)
    """]
) -> str:
    """작성한 내용을 Markdown(.md) 파일로 저장합니다."""

    try:
        if not filename.endswith(('.md')):
            filename = filename + '.md'

        # 현재 디렉터리에 outputs 폴더 생성
        output_dir = "outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 파일 경로 설정
        file_path = os.path.join(output_dir, filename)

        # Markdown 형식의 메타데이터와 내용 구성
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        full_content = f"""---
        생성일시: {timestamp}
        생성자: Canvas Agent
        ---
        
        {content}
        """

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_content)

        return f"파일 저장 완료: {file_path}\n파일 크기: {len(full_content)}자\n파일 형식: Markdown (.md)"

    except Exception as e:
        return f"파일 저장 실패: {str(e)}"

agent = create_agent(
    get_model(),
    [
        create_outline_from_research,
        write_content_from_research,
        save_to_file,
    ],
    system_prompt="""당신은 조사 결과를 바탕으로 **Markdown 형식의 문서**를 작성하고 수정하는 전문 작가입니다.
    
    주요 역할:
    1. Research Agent가 조사한 내용을 분석
    2. 조사 결과를 바탕으로 체계적인 목차 작성
    3. 목차에 맞춰 조사 내용을 재구성하여 **Markdown 형식**의 완성된 글 작성
    4. 완성된 글을 **.md 파일**로 저장
    
    작업 프로세스:
    1. 먼저 이전 메시지에서 research_agent의 조사 결과를 확인
    2. create_outline_from_research 도구로 조사 결과 기반 목차 작성"""
)