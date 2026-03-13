"""测试 LLM client 速度"""
import asyncio
import time
from config import settings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


async def test_llm():
    # enable_thinking=False 关闭 qwen3 的 thinking 模式
    llm = ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=0.7,
        model_kwargs={"extra_body": {"enable_thinking": False}},
    )

    print(f"model:    {settings.llm_model}")
    print(f"base_url: {settings.llm_base_url}")
    print("-" * 40)

    prompt = "用一句话介绍你自己"
    print(f"prompt: {prompt}")
    print("sending...")

    t0 = time.perf_counter()
    first_token_time = None
    full_text = ""

    async for chunk in llm.astream([HumanMessage(content=prompt)]):
        if first_token_time is None:
            first_token_time = time.perf_counter()
            print(f"TTFT:  {first_token_time - t0:.2f}s  (time to first token)")
        full_text += chunk.content
        print(chunk.content, end="", flush=True)

    total = time.perf_counter() - t0
    print(f"\n{'-' * 40}")
    print(f"总耗时: {total:.2f}s")
    print(f"输出:   {len(full_text)} 字符")


if __name__ == "__main__":
    asyncio.run(test_llm())
