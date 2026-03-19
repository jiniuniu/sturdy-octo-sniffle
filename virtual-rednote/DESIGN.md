# Virtual RedNote — 设计文档

## 1. 产品定位

**B端内容传播模拟工具**。品牌方输入一则内容，系统模拟它在一个 AI 驱动的虚拟社交社区中的传播过程：谁看到了、谁转发了、人们会说什么、最终是否出圈。

用户以**上帝视角**观察整个传播过程的实时演化。

---

## 2. 核心概念

### 2.1 世界（World）

模拟的基础环境，通过离线脚本 `create_world.py` 创建并持久化到 MongoDB。包含：

- 一批 AI Agent（虚拟用户），含 LLM 生成的人物 persona
- Agent 之间的有向社交关系图（networkx DiGraph）
- LLM 推导的 5~6 个语义维度 + 3~4 个人口统计维度
- 品牌 Agent 的人设（`BrandAgent`）

世界创建后保持稳定，多次模拟共用同一批 Agent。

### 2.2 维度空间（Dimension Space）

世界的核心坐标系。由 LLM 根据品牌/场景描述推导出 **5~6 个语义正交的数值维度**，每个维度有明确的低分端和高分端语义。

所有 Agent 和 Content 都用 `v ∈ [0,1]^n` 向量表示，向量决定了"偏好是否匹配"。

除数值维度外，还有 **3~4 个人口统计维度**（如职业、年龄、城市线级），用于生成多样化的 Agent persona，不参与数学计算。

### 2.3 Agent

虚拟用户，用向量 + 行为参数 + LLM 生成的 persona 描述：

```
agent.vector          # 在维度空间中的坐标 [0,1]^n，用 Sobol 序列采样
agent.tier            # kol / koc / normal
agent.activity        # 活跃度 [0,1]，影响 p_like
agent.expressiveness  # 表达欲 [0,1]，影响 p_comment
agent.sharing         # 分享倾向 [0,1]，影响 p_repost
agent.persona         # LLM 生成的自然语言人物描述，用于 LLM 评论生成
```

**Agent 采样方式（Sobol 序列）：**

不用随机采样，而用低差异 Sobol 序列 + Beta 分位数映射，保证每种 tier 的 Agent 在维度空间中均匀覆盖，避免采样偏斜：

```python
sobol = Sobol(d=n_dims, scramble=True, seed=seed)
u = sobol.random(count)                          # 均匀分布在 [0,1]^n
vectors = beta_dist.ppf(u, a=alpha, b=beta)      # 映射到 Beta 分布
```

不同 tier 使用不同 Beta 参数：KOL 偏高分端（偏好强烈），KOC 中等，Normal 均匀。

### 2.4 社交图

有向图，边表示"A 关注 B"（A 是 B 的粉丝，内容从 B 流向 A）。

构建规则（幂律分布）：
- KOL：所有 KOC/Normal 中按概率关注，粉丝数最多
- KOC：关注所有 KOL，随机关注部分 Normal
- Normal：随机关注 1~3 个 KOL/KOC

### 2.5 Content

帖子，投射到维度空间：

```
content.vector    # LLM 分析原帖得到，转发内容在此基础上加 ε ~ N(0, 0.05)
content.type      # original / repost
content.author_id # 品牌方为 "brand_001"
content.parent_id # 转发时指向源内容
```

评论不产生新 Content，不进入传播队列，只触发 LLM 异步生成文本。

### 2.6 品牌 Agent（BrandAgent）

品牌方的运营人设，包含品牌名、沟通风格、回复评论的具体方式。

在模拟过程中，每 30 个仿真事件触发一次"品牌巡查"：LLM 查看新增评论，决定是否回复（最多 3 条），不重复回复同一用户。

---

## 3. 仿真引擎

### 3.1 事件驱动架构

引擎以**优先队列（heapq）**驱动，按 `sim_time`（仿真时间，单位：天）推进：

```
EventQueue（min-heap by sim_time）
    ↓ pop
handle_exposed(event)  →  agent 决策  →  可能 push 新事件
handle_published(event) →  传播扩散  →  push 多个 exposed 事件
```

终止条件（满足任一）：
- `sim_time > time_window_days`（默认 7 天）
- `event_count > max_events`（默认 5000）
- 队列为空

### 3.2 规则引擎：行为决策

每个 `content_exposed` 事件触发一次决策：

**Step 1 — 全局去重**

```python
if agent_id in self._reached:
    return None   # 同一 agent 无论哪条内容触发，只处理一次
self._reached.add(agent_id)
```

这确保每个 Agent 在整条传播链中只被处理一次，避免重复触达。

**Step 2 — 向量匹配度**

```python
match = cosine_similarity(agent.vector, content.vector)
```

使用余弦相似度，衡量 Agent 偏好向量与内容向量的对齐程度，范围 [0,1]。

**Step 3 — 社会压力（Social Pressure）**

```python
social_pressure = Σ(weight of followers who reposted) / Σ(total weight of following)
# KOL 转发权重 = 3，其余 = 1
```

Agent 关注列表中已转发该内容的比例，KOL 转发具有更高权重，模拟"从众效应"。

**Step 4 — 行为概率采样**

```python
p_repost  = match × sharing      × (1 + social_pressure) × 0.2
p_comment = match × expressiveness × 0.3
p_like    = match × activity      × 0.6

r = random()
if   r < p_repost:                   → REPOST
elif r < p_repost + p_comment:       → COMMENT
elif r < p_repost + p_comment + p_like: → LIKE
else:                                → IGNORE
```

转发优先于评论优先于点赞——转发是最强的传播行为，优先采样。

### 3.3 传播扩散逻辑

**转发（REPOST）触发两层扩散：**

1. **直接粉丝扩散**（`_handle_published`）：内容传播给转发者的所有粉丝，时间延迟 = `0.5 / log(粉丝数 + 2)`，粉丝越多传播越快。

2. **KOL 病毒推送**（`_kol_viral_push`）：KOL 转发后立即向全局 Agent 池中最相似的 5 个未曝光 Agent 额外推送，延迟 +0.1 天。

3. **出圈扩散**（`_viral_spread`）：当累计转发数 ≥ 3 时，按内容向量相似度从全局采样额外 Agent（最多 `min(repost_count × 2, 20)` 个），模拟平台算法推荐。

**评论（COMMENT）：**

只更新计数器，不产生新 Content，不进入队列。异步触发 LLM 评论生成（fire-and-forget）。

**点赞（LIKE）：**

只更新计数器，无传播效果。

### 3.4 时间模型

仿真时间单位为"天"，每次传播的时间步长：

```python
delta_t = max(0.01, 0.5 / log(n_followers + 2))
```

粉丝越多的节点传播越快（对数关系），模拟大 V 内容的快速扩散。

### 3.5 异步 LLM 评论生成

评论文本通过 `asyncio.create_task` 异步生成，不阻塞仿真主循环：

```
agent 决策 → COMMENT
    ↓
create_task(_gen_comment)        # fire-and-forget
    ↓ （并发执行）
generate_comment(persona, post)  # LLM 调用
    ↓
comment_queue.put(result)        # 放入队列
    ↓
每个仿真步骤前 flush comment_queue → SSE 推送给前端
```

仿真结束后等待所有在途 LLM 任务完成（最多 30 秒超时）。

### 3.6 品牌巡查机制

```
每 30 个仿真事件：
    ↓
snapshot _new_comments_buf（清空缓冲）
    ↓
create_task(_run_brand_check)
    ↓
等待 2 秒（让 LLM 评论文本填充完毕）
    ↓
过滤掉已回复过的 Agent
    ↓
brand_review_comments(brand_persona, post_text, candidates)
    ↓
最多选 3 条回复，优先 KOL/KOC 和有购买意向的
    ↓
brand_queue.put(reply) → SSE brand_reply 事件
```

---

## 4. 世界生成流程（create_world.py）

```
Step 1  LLM 生成维度
        → 5~6 个数值维度（含两端语义）
        → 3~4 个人口统计维度（含标签池）
        → 品牌 Agent 人设（brand_name, tone_of_voice, reply_style）

Step 2  生成 Agent + 社交图
        → Sobol + Beta PPF 采样 Agent 向量
        → 按幂律分布构建有向社交图

Step 3  批量生成 persona（abatch 并发）
        → 每个 Agent 随机采样一组人口统计标签
        → LLM 生成 1~2 句自然语言人物描述

Step 4  持久化到 MongoDB
        → worlds（维度定义 + 品牌人设）
        → agents（向量 + persona）
        → social_graphs（边列表）
```

---

## 5. 数据模型（MongoDB）

### worlds
```json
{
  "_id": "world_abc123",
  "community_name": "成分党护肤社区",
  "description": "...",
  "created_at": "2024-01-01T00:00:00Z",
  "dimensions": [
    { "name": "成分敏感度", "description": "...", "low_label": "...", "high_label": "..." }
  ],
  "demographic_dimensions": [
    { "name": "职业类型", "labels": ["护肤博主", "皮肤科医生", "上班族"] }
  ],
  "brand_agent": {
    "brand_name": "XX品牌",
    "tone_of_voice": "专业温和",
    "reply_style": "..."
  }
}
```

### agents
```json
{
  "agent_id": "agent_0000",
  "world_id": "world_abc123",
  "tier": "kol",
  "vector": [0.92, 0.15, 0.78, 0.63, 0.44],
  "activity": 0.85,
  "expressiveness": 0.72,
  "sharing": 0.91,
  "persona": "一位常驻上海的皮肤科医生..."
}
```

复合唯一索引：`(world_id, agent_id)`。

### social_graphs
```json
{
  "_id": "world_abc123",
  "edges": [{ "s": "agent_0001", "t": "agent_0000" }]
}
```

### simulations
```json
{
  "_id": "sim_xxxxxxxxxx",
  "world_id": "world_abc123",
  "content_text": "...",
  "created_at": "...",
  "metrics": { "reach": 45, "likes": 23, "comments": 12, "reposts": 8 },
  "total_events": 312,
  "event_log": [...],
  "comments": [{ "agent_id": "agent_0012", "text": "..." }],
  "brand_replies": [{ "agent_id": "agent_0012", "reply": "..." }]
}
```

---

## 6. SSE 事件流

| type                | 内容                              | 前端行为                    |
|---------------------|-----------------------------------|-----------------------------|
| `graph_init`        | 全量节点 + 边列表                  | 初始化 D3 力导向图          |
| `agent_reacted`     | agent_id, action, metrics, sim_time | 高亮传播边 + 更新指标 + 日志 |
| `comment_generated` | agent_id, text                    | 填充评论占位卡              |
| `brand_reply`       | agent_id, reply, sim_time         | 插入品牌回复卡 + 日志       |
| `simulation_done`   | metrics, total_events             | 停止，显示保存按钮          |

---

## 7. 技术栈

| 层       | 技术                              |
|----------|-----------------------------------|
| 仿真引擎 | Python asyncio + heapq            |
| 社交图   | networkx DiGraph（内存）          |
| 采样     | scipy Sobol + Beta PPF            |
| 数据库   | MongoDB + motor（异步驱动）       |
| API      | FastAPI + sse-starlette           |
| LLM      | LangChain + OpenAI-compatible API |
| 前端     | HTML + D3.js + Chart.js           |
