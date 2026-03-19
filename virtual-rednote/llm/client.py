from langchain_openai import ChatOpenAI
from config import settings


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=temperature,
        extra_body={"enable_thinking": False},
    )
