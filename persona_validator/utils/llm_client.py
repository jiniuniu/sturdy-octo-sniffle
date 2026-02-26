from config import settings
from langchain_openai import ChatOpenAI


def get_llm(
    model: str = settings.LLM_MODEL_FAST,
    temperature: float = 0.7,
) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model=model,
        temperature=temperature,
    )
