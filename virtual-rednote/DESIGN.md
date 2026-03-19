# Virtual RedNote — 设计文档

## 1. 产品定位

**B端内容传播模拟工具**。品牌方输入一则内容，系统模拟它在一个AI驱动的虚拟社交社区中的传播过程：谁看到了、谁转发了、人们会说什么、最终是否出圈。

用户以**上帝视角**观察整个传播过程的实时演化。

---

## 2. 核心概念

### 2.1 世界（World）

模拟的基础环境，包含：
- 一批预设的 AI Agent（虚拟用户）
- Agent 之间的社交关系图
- 算法推送机制

世界在模拟开始前初始化，生命周期内保持稳定（Agent数量和关系不变，但状态会随模拟变化）。

### 2.2 维度空间（Dimension Space）

世界的核心设计。用户输入评测目标后，由 LLM 推导出 5~6 个**语义正交**的维度，构成整个模拟的坐标系。

**示例：** 评测目标 = "天然成分护肤品在年轻女性中的传播"

```
dimensions:
  - 成分敏感度    # 对天然/无添加的关注程度
  - 价格敏感度    # 对性价比的在意程度
  - 颜值导向      # 购买决策中外观权重
  - KOL依赖度     # 是否跟随博主决策
  - 环保意识      # 对可持续/绿色理念的认同
```

所有 Agent 和 Content 都在这个空间中用向量表示：`v ∈ [0,1]^n`

### 2.3 Agent

虚拟用户，用向量 + 元信息描述：

```
agent.vector      # 在维度空间中的坐标，[0,1]^n
agent.tier        # kol / koc / normal
agent.activity    # 活跃度 [0,1]，影响每步行动概率
agent.expressiveness  # 表达欲 [0,1]，影响评论概率
agent.sharing     # 分享倾向 [0,1]，影响转发概率
```

### 2.4 Content

帖子或评论，同样投射到维度空间：

```
content.vector    # 内容在维度空间中的坐标（LLM分析或规则打标）
content.author    # 发布者 agent_id 或 brand_id
content.parent_id # 转发/评论的源内容（构成传播树）
content.type      # original / repost / comment
```

### 2.5 事件（Event）

系统中所有行为都抽象为事件，驱动模拟推进：

| 事件类型 | 触发条件 | 产生结果 |
|---------|---------|---------|
| `content_published` | 品牌发帖 / agent转发 | 推送曝光给相关agent |
| `content_exposed` | 算法推送 | agent决策 |
| `agent_reacted` | agent决策非沉默 | 可能产生新的`content_published` |

---

## 3. 数据模型

### worlds
```json
{
  "_id": "world_001",
  "name": "美妆社区",
  "dimensions": [
    {"name": "成分敏感度", "description": "..."},
    {"name": "价格敏感度", "description": "..."}
  ],
  "created_at": "..."
}
```

### agents
```json
{
  "_id": "agent_001",
  "world_id": "world_001",
  "name": "小美",
  "tier": "koc",
  "vector": [0.9, 0.3, 0.7, 0.5, 0.8],
  "activity": 0.75,
  "expressiveness": 0.6,
  "sharing": 0.4,
  "following": ["agent_002", "brand_001"],
  "followers": ["agent_003", "agent_005"]
}
```

### social_graph
存储在内存（networkx DiGraph），节点为 agent_id，边带权重（关系强度）。持久化为 agents.following/followers。

### simulations
```json
{
  "_id": "sim_001",
  "world_id": "world_001",
  "brand_id": "brand_001",
  "seed_content": {
    "text": "全新天然成分精华上市...",
    "vector": [0.95, 0.4, 0.65, 0.3, 0.9]
  },
  "status": "running",
  "created_at": "...",
  "config": {
    "time_window_days": 7,
    "max_events": 10000
  }
}
```

### posts
```json
{
  "_id": "post_001",
  "sim_id": "sim_001",
  "author_id": "brand_001",
  "type": "original",
  "parent_id": null,
  "vector": [0.95, 0.4, 0.65, 0.3, 0.9],
  "text": "全新天然成分精华上市...",
  "sim_time": 0.0,
  "metrics": {"likes": 12, "comments": 3, "reposts": 2}
}
```

### events
```json
{
  "_id": "evt_001",
  "sim_id": "sim_001",
  "type": "agent_reacted",
  "sim_time": 1.2,
  "payload": {
    "agent_id": "agent_001",
    "action": "repost",
    "content_id": "post_001",
    "new_content_id": "post_002"
  }
}
```

---

## 4. 数据流转逻辑

### 4.1 世界初始化流程

```
用户输入评测目标
    ↓
LLM 推导 5~6 个正交维度
    ↓
按比例生成 agents（5% KOL, 15% KOC, 80% normal）
每个 agent 在各维度上 [0,1] 采样（按tier调整分布）
    ↓
生成社交关系图（幂律分布，KOL粉丝多）
    ↓
写入 MongoDB worlds + agents collections
构建内存 networkx 图
```

### 4.2 模拟运行流程

```
POST /simulate/start {brand_id, content_text}
    ↓
LLM 分析内容 → 生成 content.vector
写入 posts collection（type=original）
    ↓
初始化 EventQueue（优先队列，按 sim_time 排序）
推入第一批 content_exposed 事件（brand的直接粉丝）
    ↓
事件循环：
┌─────────────────────────────────────────────┐
│ event = queue.pop()                          │
│                                              │
│ if content_exposed:                          │
│   match = cosine_similarity(                 │
│     agent.vector, content.vector)            │
│   p_like    = match * agent.activity * 0.6  │
│   p_comment = match * agent.expressiveness  │
│               * 0.3                          │
│   p_repost  = match * agent.sharing         │
│               * social_pressure * 0.2        │
│   action = sample(p_like, p_comment,        │
│             p_repost, p_ignore)              │
│                                              │
│   写入 events collection                     │
│   SSE推送给前端                               │
│                                              │
│   if action in (repost, comment):            │
│     创建新 post，写入 posts                  │
│     推入 content_published 事件              │
│                                              │
│ if content_published:                        │
│   算法引擎计算曝光列表                        │
│   （粉丝 + 热度加权的非粉丝）                 │
│   推入新的 content_exposed 事件              │
│                                              │
│ 终止检查：                                   │
│   sim_time > time_window OR                  │
│   len(events) > max_events OR                │
│   queue为空                                  │
└─────────────────────────────────────────────┘
    ↓
simulation_done 事件推送，写入汇总指标
```

### 4.3 social_pressure 计算

```
agent_A 的粉丝中已转发 content_X 的比例
→ social_pressure = reposted_followers / total_followers
```

### 4.4 算法推送（content_published → 曝光列表）

```
直接粉丝（100%曝光概率）
+
非粉丝热度扩散（当 repost_count > threshold 时触发）
  → 按内容向量相似度从全局 agent 池采样
  → 采样数量 = f(repost_count)  # 出圈机制
```

### 4.5 转发内容向量

```
repost.vector = parent.vector + ε
  ε ~ Normal(0, 0.05)  # 二创引入轻微偏移
```

---

## 5. 前端展示

单页面，四个区域，SSE实时更新：

```
┌─────────────────────────────────────────────────────────────────┐
│  Virtual RedNote  [世界: 美妆社区]  [模拟进度: Day 2.3/7]         │
├──────────────────┬──────────────────┬───────────────────────────┤
│                  │                  │  实时指标                  │
│                  │  热门内容流       │  ─────────────────────    │
│  传播图           │  ──────────────  │  触达人数      ████  1243 │
│  (D3 force)      │  [头像] 小美      │  转发率        ██    12%  │
│                  │  "这个成分表真的  │  评论率        ███   18%  │
│  节点颜色:        │   太干净了！"    │  出圈指数      █     8%   │
│  品牌=红          │  ❤23 💬5 🔁12   │                           │
│  KOL=橙           │  ──────────────  │  ──────────────────────   │
│  KOC=蓝           │  [头像] KOL_陈总 │  事件日志                 │
│  normal=灰        │  "品牌方这次确实 │  ─────────────────────    │
│                  │   用心了"        │  12:03 小美 转发了原帖     │
│  边=转发/评论     │  ❤89 💬21 🔁45  │  12:03 KOL_陈总 发表评论  │
│                  │  ──────────────  │  12:04 算法推送给327人    │
│                  │  [更多...]       │  12:04 agent_233 点赞     │
│                  │                  │  12:05 二创内容出现       │
└──────────────────┴──────────────────┴───────────────────────────┘
  35% 传播图          35% 热门内容流      30% 指标 + 日志
```

**热门内容流排序逻辑：**
```
score = likes + comments * 2 + reposts * 3
```
取 Top N，每隔若干事件刷新一次（不需要每个事件都重排）。

**SSE 事件类型：**

| type | 前端行为 |
|------|---------|
| `agent_reacted` | 日志追加 + 传播图新增边 |
| `metrics_update` | 指标区更新数值 |
| `feed_update` | 热门内容流刷新 Top N 列表 |
| `simulation_done` | 停止，展示汇总 |

---

## 6. 技术栈

| 层 | 技术 |
|----|------|
| 模拟引擎 | Python asyncio + heapq |
| 社交图 | networkx（内存） |
| 数据库 | MongoDB + motor（异步驱动） |
| API | FastAPI + sse-starlette |
| 前端 | 单文件 HTML + D3.js + Chart.js |
| LLM（后期） | Anthropic Claude API |

---

## 7. 开发阶段规划

### Phase 1：规则引擎 MVP
- [ ] 世界生成（维度 + agents + 社交图）
- [ ] 事件循环核心
- [ ] 规则引擎决策（纯向量计算）
- [ ] MongoDB 数据读写
- [ ] SSE 流式推送
- [ ] 前端三区域展示

### Phase 2：LLM 增强
- [ ] 品牌内容 → 维度向量（LLM分析）
- [ ] 关键节点生成真实评论文本
- [ ] 世界生成时 LLM 推导维度

### Phase 3：产品化
- [ ] 多世界管理
- [ ] 模拟对比（同内容，不同世界参数）
- [ ] 报告导出
