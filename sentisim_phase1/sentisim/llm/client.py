"""
LLM 客户端封装 - 基于 LangChain + OpenRouter
"""

import asyncio
from typing import Optional, Type, TypeVar

from config import settings
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

T = TypeVar("T", bound=BaseModel)


class SentiSimLLM:
    """SentiSim LLM 客户端"""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.model = model or settings.default_model
        self.temperature = temperature

        # 初始化 LangChain ChatOpenAI（兼容 OpenRouter）
        self._llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )

        # 统计
        self._call_count = 0
        self._total_tokens = 0

    @property
    def call_count(self) -> int:
        """调用次数"""
        return self._call_count

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def generate(self, prompt: str) -> str:
        """
        普通文本生成

        Args:
            prompt: 提示词

        Returns:
            生成的文本
        """
        self._call_count += 1

        # LangChain 的 ainvoke 是异步的
        response = await self._llm.ainvoke(prompt)

        return response.content

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def generate_structured(
        self,
        prompt: str,
        output_schema: Type[T],
        include_format_instructions: bool = True,
    ) -> T:
        """
        结构化输出生成

        Args:
            prompt: 提示词
            output_schema: Pydantic 模型类
            include_format_instructions: 是否自动添加格式说明

        Returns:
            解析后的 Pydantic 对象
        """
        self._call_count += 1

        # 创建 Pydantic 解析器
        parser = PydanticOutputParser(pydantic_object=output_schema)

        # 构建完整 prompt
        if include_format_instructions:
            format_instructions = parser.get_format_instructions()
            full_prompt = f"{prompt}\n\n{format_instructions}"
        else:
            full_prompt = prompt

        # 调用 LLM
        response = await self._llm.ainvoke(full_prompt)

        # 解析响应
        result = parser.parse(response.content)

        return result

    async def generate_batch(
        self,
        prompts: list[str],
        max_concurrent: Optional[int] = None,
    ) -> list[str]:
        """
        批量生成（并发控制）

        Args:
            prompts: 提示词列表
            max_concurrent: 最大并发数

        Returns:
            生成结果列表
        """
        max_concurrent = max_concurrent or settings.max_concurrent_calls
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _generate_with_semaphore(prompt: str) -> str:
            async with semaphore:
                return await self.generate(prompt)

        tasks = [_generate_with_semaphore(p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        processed = []
        for r in results:
            if isinstance(r, Exception):
                processed.append(f"[ERROR] {str(r)}")
            else:
                processed.append(r)

        return processed

    async def generate_structured_batch(
        self,
        prompts: list[str],
        output_schema: Type[T],
        max_concurrent: Optional[int] = None,
    ) -> list[T | Exception]:
        """
        批量结构化生成（并发控制）

        Args:
            prompts: 提示词列表
            output_schema: Pydantic 模型类
            max_concurrent: 最大并发数

        Returns:
            解析结果列表（可能包含异常）
        """
        max_concurrent = max_concurrent or settings.max_concurrent_calls
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _generate_with_semaphore(prompt: str) -> T:
            async with semaphore:
                return await self.generate_structured(prompt, output_schema)

        tasks = [_generate_with_semaphore(p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return results


# 便捷函数：创建默认客户端
def create_llm_client(
    model: Optional[str] = None,
    temperature: float = 0.7,
) -> SentiSimLLM:
    """创建 LLM 客户端"""
    return SentiSimLLM(model=model, temperature=temperature)
