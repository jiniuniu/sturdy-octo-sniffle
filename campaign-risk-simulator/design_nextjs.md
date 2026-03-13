# Campaign Risk Simulator — 系统设计文档

## 目录

1. [背景与动机](#1-背景与动机)
2. [核心思路：从论文到产品](#2-核心思路从论文到产品)
3. [Pipeline 总览](#3-pipeline-总览)
4. [用户输入处理](#4-用户输入处理)
5. [第一步：风险维度提取](#5-第一步风险维度提取)
6. [第二步：Persona 采样与生成](#6-第二步persona-采样与生成)
7. [第三步：评论模拟](#7-第三步评论模拟)
8. [系统设计](#8-系统设计)
9. [核心数据模型](#9-核心数据模型)
10. [Zod Schemas](#10-zod-schemas)
11. [数据流转逻辑](#11-数据流转逻辑)
12. [前端布局](#12-前端布局)
13. [技术栈说明](#13-技术栈说明)

---

## 1. 背景与动机

### 问题

品牌方在发布营销活动之前，很难系统性地评估活动在不同用户群体中的潜在风险。传统的用户调研成本高、周期长，且往往只覆盖"典型用户"，忽略了边缘用户的反应——而舆情危机恰恰常常来自那些非典型的、边缘的声音。

### 目标

构建一个 Web 应用，让品牌方输入活动描述（以及可选的视觉内容描述），系统自动：

1. 提取该活动的**风险维度**（可能引发不同用户反应的轴线）
2. 生成一批**覆盖多样性**的合成用户 Persona
3. 模拟每个 Persona 对活动的**评论反应**，并标注风险信号

---

## 2. 核心思路：从论文到产品

### 参考论文

本系统的 Persona 生成逻辑参考自 Google DeepMind 的论文：

> **Persona Generators: Generating Diverse Synthetic Personas at Scale**
> Paglieri et al., 2026. arXiv:2602.03545

### 论文的关键洞察

论文提出了两个核心概念：

**Density Matching vs. Support Coverage**

传统方法（如直接让 LLM 生成"多样化用户"）倾向于生成最典型、最常见的用户画像，导致 mode collapse——所有 persona 聚集在分布中心，忽略了边缘行为。

论文的目标是 **support coverage**：覆盖可能的人群空间的完整范围，包括那些罕见但后果严重的用户。

**两阶段生成架构**

- **Stage 1（自回归）**：先在维度空间里决定每个 persona 的位置，使用 Sobol 准随机序列采样保证均匀覆盖
- **Stage 2（并行）**：基于位置描述，并行扩展每个 persona 的具体细节

### 我们的适配

| 论文原始设计                                  | 本产品适配                             |
| --------------------------------------------- | -------------------------------------- |
| 通用心理量表（Big Five、DASS 等）作为维度来源 | 从活动内容动态提取风险维度             |
| AlphaEvolve 进化优化 Persona Generator 代码   | 直接使用固定的生成逻辑（不做进化优化） |
| Concordia 多智能体模拟框架                    | 单 Persona 评论生成                    |
| Likert 量表结构化回答                         | 自由文本评论 + 风险信号标注            |
| 通用多样性评估                                | 品牌风险评估视角                       |

---

## 3. Pipeline 总览

```
用户输入活动描述 + 可选图片
        │
        ▼
  ① 处理视觉输入（可选）
  （图片 → visual_description，无图片时跳过）
        │
        ▼
  ② 提取风险维度
  （description + visual_description → 维度列表）
        │
        ▼
  ③ Sobol 采样维度空间坐标
  （纯代码，不调用 LLM）
        │
        ▼
  ④ 并行生成完整 Persona
  （8 次并行 LLM 调用，坐标直接约束输出）
        │
        ▼
  ⑤ 并行生成评论
  （8 次并行 LLM 调用，含风险信号）
        │
        ▼
  输出：风险评估报告 + 模拟评论区
```

> **关于合并 Stage 1 + Stage 2 的决策**
>
> 论文原始设计将生成拆为两阶段：Stage 1 自回归生成高层描述（保证 population 整体分布），Stage 2 并行扩展细节。在论文中 Stage 1 承担多样性控制的职责。
>
> 在本系统中，**多样性分布已由 Sobol 采样在代码层面保证**，LLM 不再需要通过自回归来协调各 persona 的位置。因此 Stage 1 退化为"把数值坐标翻译成自然语言"这一单一功能，可以直接合并进 Stage 2，每个 persona 一次调用完成，全部并行执行。
>
> 唯一的理论损失是原 Stage 1 中 LLM 可以跨 persona 做"互相参照"，但实践中这一效果有限，且 Sobol 采样已覆盖其核心作用，损失可忽略。

---

## 4. 用户输入处理

### 输入类型

| 字段          | 类型             | 说明             |
| ------------- | ---------------- | ---------------- |
| `title`       | 文字             | campaign 名称    |
| `description` | 文字             | 活动描述，必填   |
| `image`       | 图片文件（可选） | 海报、视觉物料等 |

图片是可选输入。有图片时，pipeline 在提取维度之前先将图片转为 `visualDescription` 文字，再和 `description` 一起送入后续步骤。无图片时直接跳过这一步。

### 图片上传流程

Convex Storage 原生支持文件上传，前端通过生成上传 URL 直传，不经过应用服务器：

```typescript
// components/campaign/CampaignForm.tsx

async function handleSubmit(data: FormData) {
  // 1. 创建 campaign 记录（status: "idle"）
  const campaignId = await createCampaign({
    title: data.title,
    description: data.description,
  });

  // 2. 如果有图片，上传到 Convex Storage
  if (data.image) {
    const uploadUrl = await generateUploadUrl();
    const { storageId } = await fetch(uploadUrl, {
      method: "POST",
      body: data.image,
    }).then((r) => r.json());

    // 3. 将 storageId 写回 campaign 记录
    await updateCampaignImage({ campaignId, imageStorageId: storageId });
  }

  // 4. 触发 pipeline
  await runPipeline({ campaignId });
}
```

### 图片 → visual_description

```typescript
// convex/actions/processVisual.ts

export const processVisual = action({
  args: { campaignId: v.id("campaigns") },
  handler: async (ctx, { campaignId }) => {
    const campaign = await ctx.runQuery(api.campaigns.get, { campaignId });

    // 无图片直接跳过
    if (!campaign.imageStorageId) return;

    await ctx.runMutation(api.campaigns.updateStatus, {
      campaignId,
      status: "processing_visual",
    });

    // 从 Convex Storage 读取图片，转为 base64
    const imageBlob = await ctx.storage.get(campaign.imageStorageId);
    const arrayBuffer = await imageBlob.arrayBuffer();
    const base64 = Buffer.from(arrayBuffer).toString("base64");
    const mimeType = imageBlob.type; // e.g. "image/jpeg"

    // 调用多模态 LLM
    const visualDescription = await imageChain.invoke({
      base64,
      mimeType,
      description: campaign.description,
    });

    await ctx.runMutation(api.campaigns.updateVisualDescription, {
      campaignId,
      visualDescription,
      visualStatus: "completed",
    });
  },
});
```

### imageChain（LangChain 多模态调用）

```typescript
// lib/langchain/chains/imageChain.ts

import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";

const model = new ChatOpenAI({
  modelName: "gpt-4o", // 或其他支持视觉的模型
  configuration: {
    baseURL: "https://openrouter.ai/api/v1",
    apiKey: process.env.OPENROUTER_API_KEY,
  },
});

export async function imageChain({
  base64,
  mimeType,
  description,
}: {
  base64: string;
  mimeType: string;
  description: string;
}): Promise<string> {
  const response = await model.invoke([
    new HumanMessage({
      content: [
        {
          type: "image_url",
          image_url: { url: `data:${mimeType};base64,${base64}` },
        },
        {
          type: "text",
          text: `
You are analyzing a brand campaign visual asset.
The campaign description is: "${description}"

Describe this image in detail from a brand risk analysis perspective:
- Visual composition and style (photography, illustration, color palette)
- People depicted (age, gender, appearance, clothing, activity)
- Cultural symbols, references, or iconography present
- Mood and emotional tone conveyed
- Any text, slogans, or logos visible
- Potential cultural sensitivities or ambiguities in the visual

Be specific and objective. This description will be used to identify
risk dimensions for consumer reaction simulation.
          `,
        },
      ],
    }),
  ]);

  return response.content as string;
}
```

### UI 上的原始输入展示

Campaign 详情页顶部保留用户的原始输入，让品牌方随时对照：

```
┌─────────────────────────────────────────────────────┐
│  Campaign 输入                                       │
│                                                     │
│  ┌──────────────────┐  ┌───────────────────────┐   │
│  │                  │  │ 活动描述               │   │
│  │   [原始图片]      │  │                       │   │
│  │                  │  │ 某国内新能源汽车品牌    │   │
│  │                  │  │ 发布春节营销活动。主题  │   │
│  └──────────────────┘  │ 为"新年新路，敢向前"  │   │
│                        │ ...                   │   │
│  ▼ 图片识别结果          └───────────────────────┘   │
│  ┌─────────────────────────────────────────────┐   │
│  │ 年轻女性（约25岁，短发，中性穿搭）站在空旷   │   │
│  │ 公路上，背对镜头，远处是雪山。画面比例和     │   │
│  │ 构图接近好莱坞公路片风格...   [折叠 ▲]      │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

图片通过 `useQuery(api.storage.getUrl, { storageId })` 获取临时访问 URL 渲染，`visual_description` 默认折叠，点击可展开，让用户了解 LLM 对图片的理解是否准确。

---

## 5. 第一步：风险维度提取

### 设计思路

风险维度分两层：

- **通用风险维度库**：覆盖品牌营销中常见的敏感轴线，如性别认知、文化真实性、价格阶层感知等
- **活动特异性维度**：基于当前活动内容，由 LLM 额外生成

最终输出 2-4 个维度，每个维度包含若干**语义分段（segments）**，由 LLM 自行决定分几段。分段不一定是程度递进，也可以是类型差异——比如"对活动的解读框架"天然是类型问题而非程度问题。

### Prompt

```
You are a brand risk analyst specializing in consumer sentiment and cultural dynamics.
Your task is to identify the key diversity axes along which different consumers might
react very differently to a brand campaign.

## Campaign Information
<campaign_description>
{{campaign_description}}
</campaign_description>

<visual_description>
{{visual_description}}
<!-- Leave empty if no visual content -->
</visual_description>

## Step 1: Campaign Analysis
First, briefly analyze the campaign across these lenses:
- Core message and values being communicated
- Target audience assumptions embedded in the content
- Cultural symbols, references, or imagery used
- Social issues or sensitivities potentially touched upon
- Any implicit claims (about lifestyle, identity, social status, etc.)

## Step 2: Screen Against Risk Dimension Library
Evaluate whether each of the following pre-defined dimensions is relevant to this
campaign. Mark each as [HIGH / LOW / NOT RELEVANT] with a one-line reason.

Pre-defined dimensions:
- Gender & feminism sensitivity: attitudes toward gender roles and representation
- Ethnic & cultural sensitivity: reactions to cultural symbols or stereotypes
- Environmental values: attitudes toward sustainability and corporate responsibility
- Price & class perception: sensitivity to luxury signaling or exclusivity
- Authenticity perception: belief in whether brand messaging is genuine
- Body image & health: reactions to physical appearance standards
- Nationalism & local identity: pride or resentment toward foreign vs. local brands
- Privacy & data concerns: wariness of personalization or data usage
- Political & ideological alignment: left/right value system resonance
- Generational values: attitudes that differ significantly across age groups

## Step 3: Generate Campaign-Specific Dimensions
Based on your Step 1 analysis, identify 1-3 additional dimensions that are
SPECIFIC to this campaign and not well covered by the library above.

## Step 4: Final Dimension Selection & Formatting
Select the most important 2-4 dimensions total (combining library + specific ones).
Prioritize dimensions where:
1. Different consumer types would react in fundamentally different ways
2. The campaign content has clear signals that would activate this dimension
3. The dimension captures a meaningful risk (reputational, cultural, commercial)

For each dimension, define 3-5 segments. Segments do NOT need to be ordered by
degree — they can represent genuinely distinct consumer types with different
worldviews. Each segment should be specific enough that an LLM could write a
coherent persona from it alone.

Output in the following JSON format:

{
  "dimensions": [
    {
      "id": "dim_1",
      "name": "<short name>",
      "description": "<one sentence describing what this dimension captures>",
      "relevance_reason": "<why this dimension matters for THIS specific campaign>",
      "source": "library | campaign-specific",
      "segments": [
        {
          "id": "seg_1",
          "label": "<short label, 2-4 words>",
          "description": "<one or two sentences describing this consumer type's
                          specific stance, belief, or interpretive lens>"
        }
      ]
    }
  ]
}
```

### 示例输出（新能源汽车春节活动）

```json
{
  "dimensions": [
    {
      "id": "dim_1",
      "name": "家庭责任观",
      "description": "消费者在春节团聚这件事上的核心态度与价值框架",
      "relevance_reason": "TVC 核心叙事直接围绕'春节不回家'展开，社交话题会主动放大这一分歧",
      "source": "campaign-specific",
      "segments": [
        {
          "id": "seg_1",
          "label": "传统守护者",
          "description": "春节回家是不可妥协的家庭义务，不回家等于不孝。对活动主题有本能的道德抵触。"
        },
        {
          "id": "seg_2",
          "label": "矛盾夹心层",
          "description": "内心认同家庭团聚的重要性，但能理解现实压力。看到活动会有复杂情绪，可能引发对自身处境的投射。"
        },
        {
          "id": "seg_3",
          "label": "温和个人主义者",
          "description": "认为家人之间应该互相理解，沟通比形式更重要。对活动态度中立偏正面，不会强烈反应。"
        },
        {
          "id": "seg_4",
          "label": "独立先行者",
          "description": "春节只是个节日，自我实现不应被日历绑架。对活动主题高度共鸣，可能成为自发传播者。"
        }
      ]
    },
    {
      "id": "dim_2",
      "name": "女性叙事解读",
      "description": "消费者对品牌借用女性独立议题的解读方式与情感立场",
      "relevance_reason": "主角为独自出行的年轻女性，叠加'敢'字 slogan，明显借用女性独立议题，但品牌为汽车品牌",
      "source": "library",
      "segments": [
        {
          "id": "seg_1",
          "label": "价值观共鸣者",
          "description": "真诚认同女性自我探索的叙事，将活动视为正向社会表达，愿意主动传播。"
        },
        {
          "id": "seg_2",
          "label": "审慎支持者",
          "description": "认同女性独立的价值，但对商业品牌消费这一议题保持警惕，会观察品牌是否真诚。"
        },
        {
          "id": "seg_3",
          "label": "营销洗白批判者",
          "description": "认为汽车品牌借女权话题做营销是典型的 femvertising，表面进步实则消费，感到反感。"
        },
        {
          "id": "seg_4",
          "label": "议题无感者",
          "description": "对性别议题不特别敏感，更关注产品本身或其他叙事维度，不会因此产生强烈反应。"
        }
      ]
    },
    {
      "id": "dim_3",
      "name": "文化身份认同",
      "description": "消费者对活动混用传统中国文化符号与西方美学的解读框架",
      "relevance_reason": "视觉构图接近好莱坞公路片，但配色使用传统春节元素，在国潮情绪高涨的当下可能引发真实性质疑",
      "source": "campaign-specific",
      "segments": [
        {
          "id": "seg_1",
          "label": "文化纯粹主义者",
          "description": "强调中国文化的独特性，认为春节营销混入西方美学是文化错位，对品牌真诚度产生质疑。"
        },
        {
          "id": "seg_2",
          "label": "国潮拥护者",
          "description": "热爱中国传统文化的现代表达，对视觉风格中的文化混搭感到别扭，但愿意看品牌如何自圆其说。"
        },
        {
          "id": "seg_3",
          "label": "全球化融合者",
          "description": "认为中西混搭是现代中国年轻人身份认同的自然表达，视觉风格的混搭恰恰代表了新一代消费者的自我认知。"
        }
      ]
    }
  ]
}
```

---

## 6. 第二步：Persona 采样与生成

### Sobol 采样 → Segment 映射

Sobol 序列采样出 0-1 的数值坐标，再按 segment 数量等分映射到对应的语义分段。**传给 LLM 的只有语义标签，没有任何数字。**

```typescript
// lib/sampling/sobol.ts

import { qmc } from "scipy-js"; // 或使用 sobol-seq 等纯 JS 实现

interface Segment {
  id: string;
  label: string;
  description: string;
}

interface Dimension {
  id: string;
  name: string;
  segments: Segment[];
}

function scoreToSegment(score: number, segments: Segment[]): Segment {
  const index = Math.min(
    Math.floor(score * segments.length),
    segments.length - 1,
  );
  return segments[index];
}

export function samplePersonaPositions(
  dimensions: Dimension[],
  nPersonas: number,
): Array<{
  id: string;
  // 数值保留用于 HoverCard 可视化
  scores: Record<string, number>;
  // 语义标签用于传给 LLM
  segments: Record<
    string,
    { segmentId: string; label: string; description: string }
  >;
}> {
  const sampler = new SobolSampler(dimensions.length);
  const rawPositions = sampler.sample(nPersonas);

  return rawPositions.map((pos, i) => {
    const scores: Record<string, number> = {};
    const segments: Record<
      string,
      { segmentId: string; label: string; description: string }
    > = {};

    dimensions.forEach((dim, j) => {
      const score = pos[j];
      const segment = scoreToSegment(score, dim.segments);
      scores[dim.id] = score;
      segments[dim.id] = {
        segmentId: segment.id,
        label: segment.label,
        description: segment.description,
      };
    });

    return { id: `persona_${i + 1}`, scores, segments };
  });
}
```

采样结果示例（N=8，3个维度，各有 4/4/3 个 segments）：

```
persona_1: 家庭责任观=传统守护者  女性叙事=价值观共鸣者    文化身份=文化纯粹主义者
persona_2: 家庭责任观=矛盾夹心层  女性叙事=议题无感者      文化身份=全球化融合者
persona_3: 家庭责任观=独立先行者  女性叙事=营销洗白批判者  文化身份=国潮拥护者
persona_4: 家庭责任观=温和个人主义者 女性叙事=审慎支持者   文化身份=文化纯粹主义者
persona_5: 家庭责任观=独立先行者  女性叙事=价值观共鸣者    文化身份=全球化融合者
persona_6: 家庭责任观=矛盾夹心层  女性叙事=营销洗白批判者  文化身份=国潮拥护者
persona_7: 家庭责任观=传统守护者  女性叙事=审慎支持者      文化身份=全球化融合者
persona_8: 家庭责任观=温和个人主义者 女性叙事=议题无感者   文化身份=文化纯粹主义者
```

### Persona 生成 Prompt（8 次并行调用）

```
You are generating a synthetic consumer persona for brand campaign risk evaluation.

## Campaign Context
{{campaign_description}}

## This Persona's Profile
This persona has been assigned the following position across diversity dimensions.
Each entry describes a specific consumer type — use these as hard constraints
that the persona's background, values, and life experience must naturally explain.

{{segments_for_this_persona}}
<!-- Example:
家庭责任观: 传统守护者 — 春节回家是不可妥协的家庭义务，不回家等于不孝。对活动主题有本能的道德抵触。
女性叙事解读: 审慎支持者 — 认同女性独立的价值，但对商业品牌消费这一议题保持警惕，会观察品牌是否真诚。
文化身份认同: 全球化融合者 — 认为中西混搭是现代中国年轻人身份认同的自然表达。
-->

## Instructions
Generate a complete, realistic consumer profile internally consistent with
the above segment descriptions. The persona's background and life context
should naturally explain WHY they hold these positions — not contradict them.
People can hold seemingly contradictory views; lean into that complexity.

Include:
- Basic demographics: age, gender, city, occupation, income range
- Life context: living situation, family structure, daily routine
- Values & worldview: 2-3 core beliefs shaping their consumer behavior
- Relationship with brands: how they engage with marketing content
- Psychological trigger: what specifically about this campaign
  will activate a strong reaction for them
- Identity summary: one line (age, occupation, location, one defining trait)
- Initial reaction hint: one sentence on their likely first feeling
  when seeing this campaign

Output JSON:
{
  "id": "{{persona_id}}",
  "identity_summary": "",
  "demographics": {
    "age": ,
    "gender": "",
    "city": "",
    "occupation": "",
    "income_range": ""
  },
  "life_context": "",
  "values_and_worldview": [],
  "brand_relationship": "",
  "psychological_trigger": "",
  "initial_reaction_hint": ""
}
```

### 示例输出（persona_7：传统守护者 × 审慎支持者 × 全球化融合者）

```json
{
  "id": "persona_7",
  "identity_summary": "女性，22岁，西安大学生，留守儿童背景，对家庭情感极度敏感",
  "demographics": {
    "age": 22,
    "gender": "女",
    "city": "西安（在读），老家陕西农村",
    "occupation": "大学三年级学生，兼职家教",
    "income_range": "月支出2000元以内，依赖家庭支持"
  },
  "life_context": "父母在她8岁时外出务工，由祖母抚养长大。每年春节是全家最重要的团聚节点，她从未缺席。今年是她第一次在外地过节，已经买好了回家的高铁票。关注多个女性主义账号，但在现实中仍扮演传统'乖女儿'角色。",
  "values_and_worldview": [
    "家是情感的根，离家是需要付出代价的选择，不是轻描淡写的'勇敢'",
    "女性有权追求自我，但警惕品牌把这个包装成卖点——真诚的表达和消费主义的利用是两回事",
    "中西文化融合是她生活的日常，但春节这个时刻她希望品牌能真正理解中国人的情感逻辑"
  ],
  "brand_relationship": "对品牌营销保持警惕，尤其反感用情感议题做噱头的广告。但如果内容真诚会愿意主动传播。活跃于微博和小红书，有一定表达欲。",
  "psychological_trigger": "广告里'今年不回家了，但我很好'这句文案会直接触发她的留守记忆。她既认同女主角的勇气，又对这句话的轻巧感到被冒犯——那些真正无法回家的人，背后不是'勇敢'而是无奈。",
  "initial_reaction_hint": "情绪复杂，被触动但不是感动，是被触痛，想发长文说清楚自己的不适"
}
```

---

## 7. 第三步：评论模拟

### 设计思路

借鉴论文中 Concordia 的"适当性逻辑"（Logic of Appropriateness），让每个 persona 在生成评论前先回答三个问题：

1. **这是什么情况？** → persona 如何解读这个活动
2. **我是什么样的人？** → 活动在 persona 内心激活了什么情绪
3. **像我这样的人会怎么做？** → persona 选择如何回应

输出不只是评论文本，还包括风险信号标注。

### Comment Prompt

```
You are simulating how a real consumer reacts to a brand campaign
on Chinese social media.

## Campaign Content
{{campaign_description}}

## Persona Profile
{{persona_json}}

## The Logic of Appropriateness
Before writing the comment, reason through these three questions
from the persona's perspective:

1. "What kind of situation is this?"
   → What does this persona think this campaign is trying to do?

2. "What kind of person am I?"
   → Given this persona's values and psychological trigger,
     what emotional state does this campaign put them in?

3. "What would a person like me do in this situation?"
   → Would they scroll past, leave a short comment, write a long
     post, share it, or start an argument?

## Output Format
{
  "persona_id": "{{persona_id}}",
  "reasoning": {
    "situation_reading": "...",
    "emotional_state": "...",
    "action_choice": "..."
  },
  "comment": {
    "platform": "微博 | 小红书 | 抖音评论区",
    "text": "...",
    "tone": "正面 | 负面 | 中立 | 复杂",
    "length_type": "短评(< 30字) | 中评(30-100字) | 长文(> 100字)"
  },
  "risk_signals": {
    "spread_likelihood": "低 | 中 | 高",
    "spread_reason": "...",
    "trigger_keywords": [],
    "escalation_risk": "低 | 中 | 高"
  }
}
```

### 示例输出汇总（新能源汽车春节活动）

| Persona   | 情绪 | 平台   | 传播风险 | 升级风险 | 备注          |
| --------- | ---- | ------ | -------- | -------- | ------------- |
| persona_1 | 复杂 | 小红书 | 中       | 低       |               |
| persona_2 | 中立 | 微博   | 低       | 低       |               |
| persona_3 | 负面 | 微博   | 中       | 中       | ⚠️ 不孝顺话题 |
| persona_4 | 复杂 | 小红书 | 低       | 低       |               |
| persona_5 | 正面 | 小红书 | 高       | 低       | 🔥 传播引擎   |
| persona_6 | 负面 | 微博   | 中       | 中       | ⚠️ 错误价值观 |
| persona_7 | 复杂 | 小红书 | 高       | 高       | 🚨 最大风险点 |
| persona_8 | 负面 | 微博   | 低       | 低       | 文化错位批评  |

**关键洞察**：最大风险不来自最愤怒的用户（persona_3、6），而来自情感最复杂的 persona_7——她的评论会将话题引向留守儿童这一完全不可控的社会议题。

---

## 8. 系统设计

### 项目结构

```
campaign-risk-simulator/
├── app/                          # Next.js App Router
│   ├── layout.tsx
│   ├── page.tsx                  # 重定向到第一个 campaign
│   └── campaign/
│       ├── new/
│       │   └── page.tsx          # 新建 campaign 页面
│       └── [id]/
│           └── page.tsx          # campaign 详情页（主窗口）
│
├── components/
│   ├── layout/
│   │   ├── Sidebar.tsx           # 左侧 campaign 列表
│   │   └── MainLayout.tsx        # sidebar + 主窗口框架
│   ├── campaign/
│   │   ├── CampaignForm.tsx      # 输入表单
│   │   ├── PipelineProgress.tsx  # 阶段进度展示
│   │   ├── DimensionCard.tsx     # 单个风险维度展示
│   │   ├── PersonaGrid.tsx       # persona 列表
│   │   └── PersonaCard.tsx       # 单个 persona 卡片
│   └── simulation/
│       ├── CommentFeed.tsx       # Reddit 风格评论列表
│       ├── CommentItem.tsx       # 单条评论 + hover persona
│       └── PersonaHoverCard.tsx  # hover 展示 persona 信息
│
├── convex/
│   ├── schema.ts                 # 数据模型定义
│   ├── campaigns.ts              # campaign CRUD
│   ├── pipeline.ts               # pipeline 状态更新
│   └── actions/                  # Convex actions（调用 LLM）
│       ├── processVisual.ts      # 图片 → visual_description
│       ├── extractDimensions.ts
│       ├── generatePersonas.ts
│       └── generateComments.ts
│
├── lib/
│   ├── langchain/
│   │   ├── chains/
│   │   │   ├── imageChain.ts     # 多模态图片描述
│   │   │   ├── dimensionChain.ts
│   │   │   ├── personaChain.ts
│   │   │   └── commentChain.ts
│   │   └── client.ts             # OpenRouter 配置
│   ├── sampling/
│   │   └── sobol.ts              # Sobol 序列采样
│   └── schemas/                  # Zod schemas
│       ├── dimension.ts
│       ├── persona.ts
│       └── comment.ts
│
└── types/
    └── index.ts                  # 共享类型定义
```

---

## 9. 核心数据模型

```typescript
// convex/schema.ts

import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  campaigns: defineTable({
    title: v.string(),
    description: v.string(),

    // 用户原始输入
    imageStorageId: v.optional(v.id("_storage")),

    // LLM 处理后的视觉描述
    visualDescription: v.optional(v.string()),
    visualStatus: v.optional(
      v.union(
        v.literal("pending"),
        v.literal("processing"),
        v.literal("completed"),
        v.literal("error"),
      ),
    ),

    status: v.union(
      v.literal("idle"),
      v.literal("processing_visual"),
      v.literal("extracting_dimensions"),
      v.literal("sampling"),
      v.literal("generating_personas"),
      v.literal("generating_comments"),
      v.literal("completed"),
      v.literal("error"),
    ),
    errorMessage: v.optional(v.string()),
    createdAt: v.number(),
  }),

  dimensions: defineTable({
    campaignId: v.id("campaigns"),
    dimId: v.string(), // "dim_1", "dim_2", "dim_3"
    name: v.string(),
    description: v.string(),
    relevanceReason: v.string(),
    source: v.union(v.literal("library"), v.literal("campaign-specific")),
    segments: v.array(
      v.object({
        segmentId: v.string(), // "seg_1", "seg_2", ...
        label: v.string(), // "传统守护者"
        description: v.string(), // 完整语义描述
      }),
    ),
    order: v.number(),
  }).index("by_campaign", ["campaignId"]),

  personas: defineTable({
    campaignId: v.id("campaigns"),
    personaId: v.string(), // "persona_1" ... "persona_8"
    // Sobol 采样的原始数值（保留用于 HoverCard 可视化）
    dimensionScores: v.record(
      v.string(), // dim_id
      v.number(), // 0.0 - 1.0
    ),
    // 映射后的语义 segment（传给 LLM 的实际内容）
    dimensionSegments: v.record(
      v.string(), // dim_id
      v.object({
        segmentId: v.string(), // "seg_2"
        label: v.string(), // "矛盾夹心层"
        description: v.string(), // 完整语义描述
      }),
    ),
    // 生成字段（并行写入）
    identitySummary: v.optional(v.string()),
    demographics: v.optional(
      v.object({
        age: v.number(),
        gender: v.string(),
        city: v.string(),
        occupation: v.string(),
        incomeRange: v.string(),
      }),
    ),
    lifeContext: v.optional(v.string()),
    valuesAndWorldview: v.optional(v.array(v.string())),
    brandRelationship: v.optional(v.string()),
    psychologicalTrigger: v.optional(v.string()),
    initialReactionHint: v.optional(v.string()),
    completed: v.boolean(),
  })
    .index("by_campaign", ["campaignId"])
    .index("by_campaign_persona", ["campaignId", "personaId"]),

  comments: defineTable({
    campaignId: v.id("campaigns"),
    personaId: v.string(),
    personaDbId: v.id("personas"),
    // reasoning chain
    situationReading: v.string(),
    emotionalState: v.string(),
    actionChoice: v.string(),
    // comment content
    platform: v.string(),
    text: v.string(),
    tone: v.union(
      v.literal("正面"),
      v.literal("负面"),
      v.literal("中立"),
      v.literal("复杂"),
    ),
    lengthType: v.string(),
    // risk signals
    spreadLikelihood: v.union(
      v.literal("低"),
      v.literal("中"),
      v.literal("高"),
    ),
    spreadReason: v.string(),
    triggerKeywords: v.array(v.string()),
    escalationRisk: v.union(v.literal("低"), v.literal("中"), v.literal("高")),
  })
    .index("by_campaign", ["campaignId"])
    .index("by_persona", ["personaDbId"]),
});
```

---

## 10. Zod Schemas

```typescript
// lib/schemas/dimension.ts
import { z } from "zod";

export const SegmentSchema = z.object({
  id: z.string(),
  label: z.string(),
  description: z.string(),
});

export const DimensionSchema = z.object({
  dimensions: z
    .array(
      z.object({
        id: z.string(),
        name: z.string(),
        description: z.string(),
        relevance_reason: z.string(),
        source: z.enum(["library", "campaign-specific"]),
        segments: z.array(SegmentSchema).min(3).max(5),
      }),
    )
    .min(2)
    .max(4),
});

// lib/schemas/persona.ts
export const PersonaSchema = z.object({
  id: z.string(),
  identity_summary: z.string(),
  demographics: z.object({
    age: z.number(),
    gender: z.string(),
    city: z.string(),
    occupation: z.string(),
    income_range: z.string(),
  }),
  life_context: z.string(),
  values_and_worldview: z.array(z.string()),
  brand_relationship: z.string(),
  psychological_trigger: z.string(),
  initial_reaction_hint: z.string(),
});

// lib/schemas/comment.ts
export const CommentSchema = z.object({
  persona_id: z.string(),
  reasoning: z.object({
    situation_reading: z.string(),
    emotional_state: z.string(),
    action_choice: z.string(),
  }),
  comment: z.object({
    platform: z.string(),
    text: z.string(),
    tone: z.enum(["正面", "负面", "中立", "复杂"]),
    length_type: z.string(),
  }),
  risk_signals: z.object({
    spread_likelihood: z.enum(["低", "中", "高"]),
    spread_reason: z.string(),
    trigger_keywords: z.array(z.string()),
    escalation_risk: z.enum(["低", "中", "高"]),
  }),
});
```

---

## 11. 数据流转逻辑

### 状态机

```
idle
  → processing_visual       （多模态 LLM 提取图片描述，无图片时跳过）
  → extracting_dimensions   （LLM 提取风险维度）
  → sampling                （Sobol 采样，纯代码）
  → generating_personas     （Persona LLM，8 次并行）
  → generating_comments     （Comment LLM，8 次并行）
  → completed
  → error                   （任意阶段可跳转）
```

### Convex Action 核心结构

```typescript
// convex/actions/generatePersonas.ts

export const generatePersonas = action({
  args: { campaignId: v.id("campaigns") },
  handler: async (ctx, { campaignId }) => {
    const campaign = await ctx.runQuery(api.campaigns.get, { campaignId });
    const dimensions = await ctx.runQuery(api.dimensions.getByCampaign, {
      campaignId,
    });

    // 1. Sobol 采样（纯代码）
    await ctx.runMutation(api.campaigns.updateStatus, {
      campaignId,
      status: "sampling",
    });
    const positions = samplePersonaPositions(dimensions, N_PERSONAS);
    // positions 同时包含 scores（数值）和 segments（语义标签）

    // 2. 预写入所有 persona 记录
    //    dimensionScores 用于 HoverCard 可视化
    //    dimensionSegments 用于传给 LLM
    await Promise.all(
      positions.map((pos) =>
        ctx.runMutation(api.personas.create, {
          campaignId,
          personaId: pos.id,
          dimensionScores: pos.scores,
          dimensionSegments: pos.segments,
          completed: false,
        }),
      ),
    );

    await ctx.runMutation(api.campaigns.updateStatus, {
      campaignId,
      status: "generating_personas",
    });

    // 3. 并行生成，传入语义 segments 而非数值
    await Promise.all(
      positions.map(async (pos) => {
        // 将 segments 格式化为自然语言传给 LLM
        const segmentsText = Object.entries(pos.segments)
          .map(([dimId, seg]) => {
            const dim = dimensions.find((d) => d.dimId === dimId);
            return `${dim.name}: ${seg.label} — ${seg.description}`;
          })
          .join("\n");

        const result = await personaChain.invoke({
          campaign,
          segmentsText,
          personaId: pos.id,
        });

        const parsed = PersonaSchema.parse(result);

        await ctx.runMutation(api.personas.update, {
          campaignId,
          personaId: pos.id,
          ...parsed,
          completed: true,
        });
      }),
    );

    await ctx.runMutation(api.campaigns.updateStatus, {
      campaignId,
      status: "generating_comments",
    });
  },
});
```

### 实时性说明

Convex 的响应式查询保证：

- 维度写入后，前端立刻渲染维度卡片，无需等待 persona 生成
- Persona 记录在采样后立即预写入（`completed: false`），前端立刻显示 8 个"生成中"占位卡片
- 每个 persona 生成完成后（`completed: true`），前端立刻填充对应卡片内容
- 每条评论写入后，前端立刻追加到评论列表

---

## 12. 前端布局

### 整体结构

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────────┐  ┌─────────────────────────────────────┐ │
│  │   Sidebar    │  │           Main Window               │ │
│  │              │  │                                     │ │
│  │ + New        │  │  ┌─────────────────────────────┐   │ │
│  │              │  │  │     Pipeline Progress        │   │ │
│  │ ──────────   │  │  │                             │   │ │
│  │              │  │  │  ① 提取维度      ✓          │   │ │
│  │ ● 新能源春节  │  │  │  ② 生成Persona  ████░░ 5/8 │   │ │
│  │   活动       │  │  │  ③ 模拟评论     waiting...  │   │ │
│  │              │  │  └─────────────────────────────┘   │ │
│  │ ○ 某美妆品牌  │  │                                     │ │
│  │   活动       │  │  ┌─────────────────────────────┐   │ │
│  │              │  │  │     风险维度                  │   │ │
│  │              │  │  │  [家庭责任观] [性别角色认知]  │   │ │
│  │              │  │  │  [文化真实性感知]             │   │ │
│  │              │  │  └─────────────────────────────┘   │ │
│  │              │  │                                     │ │
│  │              │  │  ┌─────────────────────────────┐   │ │
│  │              │  │  │     模拟评论区                │   │ │
│  │              │  │  │                             │   │ │
│  │              │  │  │  ┌──────────────────────┐   │   │ │
│  │              │  │  │  │ [头像] persona_5      │   │   │ │
│  │              │  │  │  │ ★★★★★  小红书  正面   │   │   │ │
│  │              │  │  │  │ 看哭了。每年春节被问... │   │   │ │
│  │              │  │  │  │ 🔥传播:高  ⚡升级:低   │   │   │ │
│  │              │  │  │  └──────────────────────┘   │   │ │
│  │              │  │  │                             │   │ │
│  │              │  │  │  ┌──────────────────────┐   │   │ │
│  │              │  │  │  │ [头像] persona_7      │   │   │ │
│  │              │  │  │  │ 😰  小红书  复杂       │   │   │ │
│  │              │  │  │  │ 说说我为什么看这个...  │   │   │ │
│  │              │  │  │  │ 🔥传播:高  ⚡升级:高   │   │   │ │
│  │              │  │  │  └──────────────────────┘   │   │ │
│  │              │  │  └─────────────────────────────┘   │ │
│  └──────────────┘  └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### PersonaHoverCard

当用户 hover 到评论的头像时，弹出该 persona 的完整信息。维度展示使用语义 segment 标签，不再使用进度条：

```
┌──────────────────────────────────┐
│  persona_7                       │
│  女，22岁，西安大学生             │
│  ──────────────────────────      │
│  家庭责任观                      │
│  ◆ 传统守护者                    │
│  春节回家是不可妥协的义务...      │
│                                  │
│  女性叙事解读                    │
│  ◆ 审慎支持者                    │
│  认同独立价值，但警惕品牌消费...  │
│                                  │
│  文化身份认同                    │
│  ◆ 全球化融合者                  │
│  中西混搭是现代年轻人的表达...    │
│  ──────────────────────────      │
│  "父母在她8岁时外出务工，         │
│   由祖母抚养长大..."              │
│  ──────────────────────────      │
│  心理触发点：                     │
│  "广告里'今年不回家了，           │
│   但我很好'会直接触发             │
│   她的留守记忆..."                │
└──────────────────────────────────┘
```

### 评论排序逻辑

默认按 `escalation_risk` 降序排列（高风险评论置顶），让品牌方第一眼看到最需要关注的内容。可提供切换：按风险排序 / 按传播可能性排序 / 按情绪分类查看。

---

## 13. 技术栈说明

| 层级       | 技术选型                            | 原因                                                                   |
| ---------- | ----------------------------------- | ---------------------------------------------------------------------- |
| 前端框架   | Next.js (App Router)                | 与 Convex 配合成熟                                                     |
| 实时数据   | Convex                              | 响应式查询让 pipeline 中间状态自然实时推送到前端，无需手写 WebSocket   |
| 文件存储   | Convex Storage                      | 原生集成，图片直传不经过应用服务器，storageId 直接存在 campaign 记录里 |
| LLM 调用   | LangChain + OpenRouter              | LangChain 提供结构化输出封装，OpenRouter 允许灵活切换模型              |
| 多模态调用 | LangChain HumanMessage（image_url） | 图片以 base64 传入，兼容所有支持视觉的模型                             |
| 输出验证   | Zod                                 | 与 LangChain 的 `withStructuredOutput` 集成，确保 LLM 输出格式符合预期 |
| 采样       | Sobol 序列（scipy / 纯 JS 实现）    | 准随机序列保证维度空间的均匀覆盖，这是避免 persona 聚集的关键          |
| 样式       | Tailwind CSS                        | 极简 UI 快速开发                                                       |

### 关键设计决策

**为什么用语义 segment 而不是数值传给 LLM**

LLM 对数值的感知很弱，`0.18` 和 `0.25` 在语义上几乎没有区别。将 Sobol 采样的数值在代码层面先映射为语义标签，LLM 直接消费完整的自然语言描述，生成的 persona 会更准确地体现目标类型。数值本身保留在 `dimensionScores` 字段，仅用于内部追踪，不传给任何 LLM。

**为什么 segment 数量由 LLM 自行决定**

不同维度天然有不同的分段逻辑。有些维度是程度问题（3-4 段够了），有些是类型问题（可能需要 4-5 个差异显著的类型）。固定分段数会强迫 LLM 在不合适的地方做切割，反而降低 segment 的语义质量。Zod schema 约束在 3-5 段之间作为合理范围。

**为什么并行生成用预写入占位记录**

采样完成后立即写入 N 条 `completed: false` 的 persona 记录，前端订阅后立刻渲染 8 个占位卡片。用户能清楚看到"正在生成第几个"，而不是等所有完成后一次性出现。

**为什么 dimensions / personas / comments 分开存表**

分表后前端可以分别订阅，维度出来后立即渲染，不需要等待所有 persona 完成。每张表的 `by_campaign` 索引保证查询效率。

**为什么 PersonaHoverCard 不需要额外请求**

前端订阅了整个 campaign 的 personas 表，所有 persona 数据已在本地，hover 时直接读取，零延迟。

---

_文档版本：v1.0 | 最后更新：2026-02_
