from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from chains.client import get_llm
from models.dimension import DimensionsOutput
from models.study import StudyDesign

_PROMPT = """你是一名专注于消费者行为研究的分析师。
你的任务是识别不同消费者对同一研究对象可能产生截然不同反应的关键差异轴线（研究维度）。

## 研究信息

<研究目标>
{research_objective}
</研究目标>

<研究对象描述>
{stimulus_description}
</研究对象描述>

<视觉内容描述>
{visual_description}
</视觉内容描述>

## 第一步：研究对象分析
从以下角度简要分析这个研究对象：
- 传递的核心信息与价值主张
- 内容中隐含的目标受众假设
- 使用的文化符号、意象或参考
- 可能触及的社会议题或敏感点
- 任何隐性主张（关于生活方式、身份认同、社会地位等）

## 第二步：对照维度库筛查
评估以下预定义维度与本研究对象的相关性，每项标注 [高度相关 / 低度相关 / 不相关] 并附一句理由：

预定义维度：
- 性别与女性主义敏感度：对性别角色与表达方式的态度
- 民族与文化敏感度：对文化符号或刻板印象的反应
- 环保价值观：对可持续发展与企业责任的态度
- 价格与阶层感知：对奢侈品信号或排他性的敏感度
- 真实性感知：对品牌信息是否真诚的判断
- 身体形象与健康：对外貌标准的反应
- 民族主义与本土认同：对外资与本土品牌的自豪感或抵触感
- 隐私与数据顾虑：对个性化推荐或数据使用的警惕
- 政治与意识形态立场：左右价值体系的共鸣程度
- 代际价值观：不同年龄段之间显著差异的态度

## 第三步：生成研究特异性维度
基于第一步分析，识别 1-3 个本研究特有的、上述维度库未能覆盖的差异维度。
**重点**：维度选择应服务于研究目标「{research_objective}」，优先选择与该目标最相关的差异轴线。

{preset_dimensions_section}

## 第四步：最终维度选择与格式化
综合维度库与特异性维度，选出最重要的 2-4 个。优先选择：
1. 不同消费者类型会产生根本性不同反应的维度
2. 研究对象内容有明确信号会激活这一维度
3. 该维度捕捉到与研究目标直接相关的差异

每个维度定义 3-5 个分段。分段不必按程度递进，可以是持有不同世界观的消费者类型。
每个分段的描述应足够具体，单独看这段描述就能写出一个连贯的 persona。

**关于 segment label 的写法**：
label 应直接描述这类消费者的核心立场或态度，用短语而非名词标签。
例如："春节必须回家才是家人" 而非 "传统守护者"；"理解但内心矛盾" 而非 "矛盾夹心层"。

## 第五步：隐藏系统维度（必须执行）
在上述 2-4 个可见维度之后，必须在数组末尾追加恰好 2 个隐藏维度，source 设为 "hidden"。
这两个维度不展示给用户，仅用于确保 persona 的地域和职业多样性。

隐藏维度 1 — id 固定为 "dim_region"：
  目标市场为：{target_market}
  生成 4-5 个分段，代表该市场内有意义的地域/社会经济层次。
  参考指南（按市场）：
  - 中国大陆：一线城市（北京/上海/广州/深圳）、新一线城市（成都/杭州/武汉等）、二线城市、三四线城市、县城及农村
  - 台湾：台北都会区、台中与高雄、中小城市、乡镇及东部地区
  - 日本：东京/大阪都市圈、地方中心城市（名古屋/札幌/福冈）、中小城市、农村地区
  - 韩国：首尔/京畿道、其他主要城市（釜山/仁川/大邱）、中小城市、农村地区
  - 东南亚：主要首都都市圈、二线城市、省级城镇、农村地区
  - 印度：顶级都市（孟买/德里/班加罗尔/海德拉巴）、二线城市、三线城市及城镇、农村
  - 美国：主要沿海都市圈、阳光地带城市、中西部/南部中等城市、农村及小城镇
  请严格对应上方声明的目标市场生成分段。

隐藏维度 2 — id 固定为 "dim_occupation"：
  目标市场为：{target_market}
  生成 4-5 个分段，代表该市场中不同职业或人生阶段的消费者原型。
  参考指南（按市场）：
  - 中国大陆：白领/职场专业人士、蓝领/体力劳动者、自由职业者及创业者、学生、全职主妇/退休人员
  - 日本/韩国：企业员工（上班族）、小企业主、学生、全职主妇、退休人员
  - 东南亚/印度：正规部门专业人士、非正规/零工经济从业者、小企业主、学生、全职主妇
  - 欧美：知识工作者、服务业及技术工人、创业者及自由职业者、学生、退休人员
  请严格对应上方声明的目标市场生成分段。

目标市场：{target_market}

{format_instructions}
"""

_PRESET_DIMENSIONS_SECTION = """## 用户预设维度（必须包含）
以下维度由研究者指定，必须原样包含在最终输出中（source 设为 "preset"），再从剩余空间补充其他维度：
{preset_list}
"""


def _build_preset_section(preset_dimensions: list) -> str:
    if not preset_dimensions:
        return ""
    lines = []
    for i, d in enumerate(preset_dimensions, 1):
        lines.append(f"{i}. 【{d.name}】{d.description}")
        if d.segments:
            lines.append(f"   预设分段：{'、'.join(d.segments)}")
    return _PRESET_DIMENSIONS_SECTION.format(preset_list="\n".join(lines))


async def extract_dimensions(
    stimulus_description: str,
    visual_description: str = "",
    target_market: str = "中国大陆",
    study_design: StudyDesign | None = None,
) -> DimensionsOutput:
    parser = PydanticOutputParser(pydantic_object=DimensionsOutput)
    llm = get_llm(temperature=0.3)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=get_llm(temperature=0))

    research_objective = (
        study_design.research_objective
        if study_design
        else "识别消费者对该内容可能产生的不同反应"
    )
    preset_dimensions = study_design.preset_dimensions if study_design else []

    response = await llm.ainvoke(
        _PROMPT.format(
            research_objective=research_objective,
            stimulus_description=stimulus_description,
            visual_description=visual_description or "（无视觉内容）",
            target_market=target_market,
            preset_dimensions_section=_build_preset_section(preset_dimensions),
            format_instructions=parser.get_format_instructions(),
        )
    )
    try:
        return parser.parse(response.content)
    except Exception:
        return fixing_parser.parse(response.content)
