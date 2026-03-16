#!/usr/bin/env python3
"""
测试 dimension_chain 的原始输出，打印：
1. 发给 LLM 的完整 prompt（token 量参考）
2. LLM 返回的原始 response.content（看中间分析有多长）
3. 解析后的结构化结果（维度数量、分段数量）
4. 耗时
"""
import asyncio
import time
import sys

# 需要在项目根目录运行，确保能 import 本地模块
from chains.client import get_llm
from langchain_core.output_parsers import PydanticOutputParser
from models.dimension import DimensionsOutput
from models.study import StudyDesign

# ── 测试用例（改这里来测不同场景） ────────────────────────────────
STIMULUS_DESCRIPTION = """
三八妇女节营销活动方案：
主题："女性独立靠自己，买包不靠男人"
目标受众：25-40岁都市女性
投放渠道：微博、小红书
核心创意：展示不同职业女性用自己收入购买奢侈品的场景，配合"自己赚的，凭什么不买"文案
"""

STUDY_DESIGN = StudyDesign(
    study_type="risk_assessment",
    research_objective="识别该营销活动可能引发的舆论风险和消费者负面反应",
    response_mode="comment",
    analysis_framework="risk",
)

TARGET_MARKET = "中国大陆"


async def main():
    from chains.dimension_chain import _PROMPT, _build_preset_section
    from langchain_core.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=DimensionsOutput)
    llm = get_llm(temperature=0.3)

    research_objective = STUDY_DESIGN.research_objective
    preset_dimensions = STUDY_DESIGN.preset_dimensions if STUDY_DESIGN else []

    full_prompt = _PROMPT.format(
        research_objective=research_objective,
        stimulus_description=STIMULUS_DESCRIPTION,
        visual_description="（无视觉内容）",
        target_market=TARGET_MARKET,
        preset_dimensions_section=_build_preset_section(preset_dimensions),
        format_instructions=parser.get_format_instructions(),
    )

    # ── 打印 prompt ──────────────────────────────────────────────
    print("=" * 60)
    print("PROMPT（发给 LLM 的完整内容）")
    print("=" * 60)
    print(full_prompt)
    print(f"\n[prompt 字符数: {len(full_prompt)}]")

    # ── 调用 LLM ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("调用 LLM 中...")
    print("=" * 60)
    t0 = time.time()
    response = await llm.ainvoke(full_prompt)
    elapsed = time.time() - t0

    # ── 打印原始响应 ──────────────────────────────────────────────
    print(f"\n[耗时: {elapsed:.1f}s]")
    print(f"[响应字符数: {len(response.content)}]")
    print("\n" + "=" * 60)
    print("LLM 原始响应（response.content）")
    print("=" * 60)
    print(response.content)

    # ── 解析结果 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("解析结果")
    print("=" * 60)
    try:
        result = parser.parse(response.content)
        print(f"维度数量: {len(result.dimensions)}")
        for d in result.dimensions:
            print(f"  [{d.source}] {d.id}: {d.name}（{len(d.segments)} 个分段）")
            for s in d.segments:
                print(f"    - {s.label}")
    except Exception as e:
        print(f"解析失败: {e}")
        print("原始内容见上方")


if __name__ == "__main__":
    asyncio.run(main())
