# SentiSim 舆情风险推演系统设计文档

## 基于LLM的社交网络舆情模拟与风险预测系统

**版本**: v2.1  
**日期**: 2026年1月

---

## 一、设计理念

### 1.1 核心原则

传统的智能体模拟依赖大量手工规则（影响力公式、学习率、阈值等），这些规则难以捕捉人类行为的复杂性和多样性。本系统采用**纯LLM驱动**的方案：

- 所有"理解"和"判断"都交给LLM
- 代码只负责流程编排和状态记录
- 用自然语言描述替代数值参数

### 1.2 系统定位

```
输入                        处理                      输出
────                        ────                      ────
品牌背景描述          →     生成虚拟社会        →    风险评分
目标受众描述          →     模拟传播过程        →    传播预测
待测营销内容          →     （全程LLM驱动）     →    负面反应样本
                                                     优化建议
```

---

## 二、数据模型

### 2.1 品牌配置

```
BrandContext
├── brand_name: str           # 品牌名称
├── description: str          # 品牌背景描述（一段话，包含行业、调性、历史等）
├── target_audience: str      # 目标受众描述（一段话）
└── sensitivity_topics: list  # 需要关注的敏感议题列表
```

**示例**：

```python
brand_context = {
    "brand_name": "XX饮料",
    "description": """
        XX饮料是一家主打健康概念的饮品品牌，目标客群为18-35岁注重生活品质的年轻人。
        品牌调性偏年轻、活力、健康。
        历史上曾在2024年因代糖成分问题引发争议，后官方澄清但部分消费者仍有疑虑。
        去年有过一次涨价，引发部分老用户不满。
    """,
    "target_audience": """
        主要是一二线城市的年轻白领和大学生，关注健康、热爱社交媒体，
        对食品安全和成分比较敏感，价格敏感度中等，容易被KOL种草也容易被负面新闻劝退。
    """,
    "sensitivity_topics": ["食品安全", "添加剂/代糖", "价格", "虚假宣传"]
}
```

### 2.2 人群类型

```
PersonaType
├── type_name: str        # 人群名称
├── description: str      # 人群完整描述（一段话，包含特征、态度、行为倾向等）
└── proportion: float     # 在目标受众中的占比
```

**说明**：不拆分为多个子结构，全部用自然语言描述。

### 2.3 用户

```
User
├── user_id: str
├── persona_type: str      # 所属人群类型名称
├── profile: str           # 用户画像描述（一段话，由LLM生成）
├── influence_level: str   # "kol" | "active" | "normal" | "silent"（由网络结构决定）
├── memory: list[str]      # 记忆列表（自然语言条目）
└── follower_ids: list     # 粉丝ID列表（由网络结构决定）
```

### 2.4 帖子

```
Post
├── post_id: str
├── author_id: str
├── content: str           # 帖子内容
├── post_type: str         # "brand_original" | "forward" | "forward_comment" | "creation"
├── original_post_id: str  # 原帖ID（如果是转发/二创）
└── timestamp: int
```

### 2.5 社交网络

```
SocialNetwork
├── graph: Graph           # 网络拓扑结构
├── users: dict[str, User] # 用户字典
└── brand_account_id: str  # 品牌官方账号ID
```

---

## 三、核心流程

### 3.1 整体流程

```
┌─────────────────┐
│ 1. 生成人群类型  │ ◀── LLM
└────────┬────────┘
         ▼
┌─────────────────┐
│ 2. 构建网络拓扑  │ ◀── 算法（BA无标度网络）
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3. 分配影响力角色│ ◀── 规则（基于节点度数）
└────────┬────────┘
         ▼
┌─────────────────┐
│ 4. 生成用户画像  │ ◀── LLM（结合角色和人群类型）
└────────┬────────┘
         ▼
┌─────────────────┐
│ 5. 初始化用户记忆│ ◀── LLM（基于品牌历史）
└────────┬────────┘
         ▼
┌─────────────────┐
│ 6. 注入待测内容  │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 7. 传播模拟循环  │ ◀── LLM（每个用户一次调用）
└────────┬────────┘
         ▼
┌─────────────────┐
│ 8. 生成风险报告  │ ◀── LLM
└─────────────────┘
```

### 3.2 传播模拟循环

```
当前传播队列: [post1, post2, ...]

FOR 每条帖子 IN 队列:
    FOR 每个能看到帖子的用户:
        │
        ▼
    ┌────────────────────────────┐
    │ LLM一次调用：               │
    │   输入：用户画像 + 记忆 + 帖子 │
    │   输出：反应 + 行动 + 新内容   │
    └────────────────────────────┘
        │
        ▼
    ┌────────────────────────────┐
    │ 状态更新：                  │
    │   - 更新记忆               │
    │   - 新内容加入下一轮队列     │
    └────────────────────────────┘
```

---

## 四、核心算法

### 4.1 生成人群类型

```
FUNCTION generate_persona_types(brand_context) -> list[PersonaType]:

    prompt = f"""
    【品牌背景】
    {brand_context.description}

    【目标受众】
    {brand_context.target_audience}

    【敏感议题】
    {brand_context.sensitivity_topics}

    请为该品牌生成6-8种典型的受众人群分类。

    要求：
    1. 覆盖对品牌态度从正面到负面的不同人群
    2. 各人群比例之和为100%
    3. 每个人群用一段话完整描述其特征、态度、敏感点、行为倾向

    注意：不需要考虑影响力层级（KOL/普通用户等），这会在后续步骤中根据网络结构分配。

    输出格式（JSON）：
    [
      {{
        "type_name": "人群名称",
        "description": "完整描述...",
        "proportion": 15
      }},
      ...
    ]
    """

    response = LLM.call(prompt)
    persona_types = parse_json(response)

    RETURN persona_types
```

### 4.2 构建网络拓扑

使用经典的**BA无标度网络模型**（Barabási-Albert），它能生成符合真实社交网络特征的拓扑结构：少数节点有大量连接（KOL），多数节点连接较少（普通用户）。

```
FUNCTION build_network_topology(user_count) -> Graph:

    # 使用BA模型生成无标度网络
    # 参数m：每个新节点加入时连接的边数
    graph = ba_model(n=user_count, m=3)

    # BA模型生成的是无向图，转换为有向图（关注关系）
    directed_graph = convert_to_directed(graph)

    # 添加品牌官方账号节点
    brand_node = add_brand_node(directed_graph)

    # 大部分用户关注品牌账号
    FOR node IN random_sample(directed_graph.nodes, ratio=0.7):
        add_edge(node, brand_node)

    RETURN directed_graph
```

**为什么选择BA模型**：

- 生成的网络符合幂律分布（少数超级节点，多数普通节点）
- 与真实社交网络的拓扑特征相似
- 算法简单高效，适合大规模生成

### 4.3 分配影响力角色

根据网络中节点的**入度**（粉丝数）来分配角色，让角色从网络结构中自然涌现：

```
FUNCTION assign_influence_levels(graph) -> dict:

    # 计算每个节点的入度（被关注数）
    in_degrees = {}
    FOR node IN graph.nodes:
        in_degrees[node.id] = count_incoming_edges(graph, node)

    # 按入度降序排列
    sorted_nodes = sort_by_value(in_degrees, descending=True)
    total = len(sorted_nodes)

    # 根据排名分配角色
    influence_levels = {}
    FOR rank, (node_id, degree) IN enumerate(sorted_nodes):
        percentile = rank / total

        IF percentile < 0.03:           # 前3%：KOL
            influence_levels[node_id] = "kol"
        ELSE IF percentile < 0.15:      # 3%-15%：活跃用户
            influence_levels[node_id] = "active"
        ELSE IF percentile < 0.60:      # 15%-60%：普通用户
            influence_levels[node_id] = "normal"
        ELSE:                           # 后40%：沉默用户
            influence_levels[node_id] = "silent"

    RETURN influence_levels
```

**角色分布**：

| 角色     | 比例 | 特征                       |
| -------- | ---- | -------------------------- |
| KOL      | ~3%  | 入度最高，粉丝量大         |
| 活跃用户 | ~12% | 入度较高，有一定影响力     |
| 普通用户 | ~45% | 入度中等                   |
| 沉默用户 | ~40% | 入度最低，主要是信息接收者 |

### 4.4 生成用户画像

结合**网络决定的角色**和**随机分配的人群类型**来生成具体用户：

```
FUNCTION generate_users(graph, influence_levels, persona_types) -> list[User]:

    users = []

    FOR node IN graph.nodes:

        # 获取该节点的影响力角色
        influence_level = influence_levels[node.id]

        # 按比例随机选择一个人群类型
        persona = weighted_random_choice(persona_types, weights=[p.proportion for p in persona_types])

        # 获取粉丝数（入度）
        follower_count = count_incoming_edges(graph, node)
        follower_ids = get_incoming_nodes(graph, node)

        # LLM生成用户画像
        prompt = f"""
        【人群类型】
        {persona.type_name}

        【人群特征】
        {persona.description}

        【该用户的影响力】
        影响力层级：{influence_level}
        粉丝数量：约{follower_count}人

        请生成一个具体用户的画像描述（80-120字），包含：
        - 年龄、性别、职业、所在城市
        - 性格特点和表达风格
        - 与该品牌的具体故事或关系
        - 如果是KOL，说明其影响力领域

        要求画像具体生动，有个人特色。
        直接输出描述，不要JSON格式。
        """

        profile = LLM.call(prompt)

        user = User(
            user_id = node.id,
            persona_type = persona.type_name,
            profile = profile,
            influence_level = influence_level,
            memory = [],
            follower_ids = follower_ids
        )
        users.append(user)

    RETURN users
```

**示例用户画像**：

KOL示例：

```
32岁男性，上海的美食博主，全网粉丝50万+，以"成分党测评"闻名。
说话直接犀利，喜欢用数据说话。曾经测评过XX饮料，当时给了中评，
指出代糖问题后引发品牌回应。粉丝信任度高，一条测评能带动明显销量波动。
对品牌持观望态度，愿意给机会但不会轻易背书。
```

普通用户示例：

```
24岁女生，杭州互联网公司的运营，typical的都市打工人。
日常靠奶茶续命，最近在尝试戒糖所以开始喝无糖饮料。
XX饮料是最近才开始买的，觉得还行但没有特别忠诚。
刷到相关内容会看一眼，偶尔点赞，很少评论，从不转发。
```

### 4.5 初始化用户记忆

```
FUNCTION initialize_memories(users, brand_context):

    FOR user IN users:

        prompt = f"""
        【用户画像】
        {user.profile}

        【品牌背景】
        {brand_context.description}

        基于该用户的特点和品牌的历史，这个用户对该品牌可能有什么记忆或印象？

        请生成1-3条记忆，每条用一句话表达。
        如果这类用户可能没有特别的记忆，可以只返回1条或返回空列表。

        输出格式（JSON字符串数组）：
        ["记忆1", "记忆2"]
        """

        response = LLM.call(prompt)
        user.memory = parse_json(response)
```

### 4.6 用户反应模拟（核心）

**一次LLM调用完成全部判断**：感知 → 情绪 → 印象变化 → 行动决策 → 内容生成

```
FUNCTION simulate_user_response(user, post, post_author) -> Response:

    # 构建记忆上下文
    IF user.memory:
        memory_text = "\n".join([f"- {m}" for m in user.memory])
    ELSE:
        memory_text = "（暂无特别记忆）"

    # 构建发布者描述
    IF post_author:
        author_desc = f"一位用户：{post_author.profile[:100]}..."
    ELSE:
        author_desc = "品牌官方账号"

    prompt = f"""
    【你的身份】
    {user.profile}

    【你对该品牌的记忆】
    {memory_text}

    【你刚刚看到的内容】
    发布者：{author_desc}
    内容："{post.content}"

    ---

    请完全代入这个用户的身份，回答以下问题：

    1. **第一反应**：看到这条内容，你的第一反应是什么？（一句话）

    2. **情绪感受**：你现在的情绪是怎样的？（如：开心、无感、反感、愤怒、担忧、想吐槽等）

    3. **印象变化**：这条内容会改变你对该品牌的印象吗？如果会，怎么变？（一句话描述变化，如果没变化就回答"没有变化"）

    4. **行动决定**：你会怎么做？选择一个：
       - ignore: 划走，不互动
       - like: 点赞
       - forward: 直接转发
       - forward_comment: 转发并写一句评论
       - create: 发一条自己的内容

    5. **发布内容**：如果选择了forward_comment或create，你会写什么？（否则留空）

    ---

    输出格式（JSON）：
    {{
      "first_reaction": "...",
      "emotion": "...",
      "impression_change": "...",
      "action": "ignore|like|forward|forward_comment|create",
      "content": "..."
    }}
    """

    response = LLM.call(prompt)
    result = parse_json(response)

    RETURN result
```

### 4.7 记忆更新机制

记忆更新发生在每次用户反应之后，遵循以下原则：

**更新时机**：只有当用户的印象发生变化时才记录

**更新内容**：记录印象变化本身，而不是看到的内容

**长度限制**：保留最近N条，简单截断

```
FUNCTION update_user_memory(user, response):

    # 只有印象发生变化时才记录
    impression_change = response["impression_change"]

    IF impression_change AND impression_change != "没有变化":
        user.memory.append(impression_change)

    # 保持记忆长度在合理范围
    MAX_MEMORY_SIZE = 10
    IF len(user.memory) > MAX_MEMORY_SIZE:
        user.memory = user.memory[-MAX_MEMORY_SIZE:]
```

**设计说明**：

| 设计选择        | 原因                                 |
| --------------- | ------------------------------------ |
| 只记"印象变化"  | 不是所有内容都值得记住，只记有影响的 |
| LLM决定是否变化 | 不用规则判断"重要性"，让LLM自己说    |
| 简单追加+截断   | 避免复杂的重要性排序算法             |
| 保留最近的      | 近因效应，最近的记忆影响更大         |

**记忆演化示例**：

```
初始记忆：
["去年代糖事件时有点担心，后来看官方澄清继续买了"]

看到品牌内容后，LLM输出：
{
  "impression_change": "感觉品牌还是在回避代糖问题，有点失望"
}

更新后记忆：
["去年代糖事件时有点担心，后来看官方澄清继续买了",
 "感觉品牌还是在回避代糖问题，有点失望"]

又看到一条正面测评后：
{
  "impression_change": "看到博主的详细测评，原来成分确实没问题，之前误解了"
}

更新后记忆：
["去年代糖事件时有点担心，后来看官方澄清继续买了",
 "感觉品牌还是在回避代糖问题，有点失望",
 "看到博主的详细测评，原来成分确实没问题，之前误解了"]
```

### 4.8 传播模拟主循环

```
FUNCTION run_simulation(network, test_content, max_steps) -> SimulationResult:

    # 创建品牌初始帖子
    brand_post = Post(
        post_id = generate_id(),
        author_id = network.brand_account_id,
        content = test_content,
        post_type = "brand_original",
        timestamp = 0
    )

    all_posts = [brand_post]
    all_responses = []
    current_wave = [brand_post]

    FOR step IN range(max_steps):

        next_wave = []

        FOR post IN current_wave:

            # 获取能看到这条帖子的用户
            author = network.get_user(post.author_id)
            audience_ids = author.follower_ids

            FOR user_id IN audience_ids:

                user = network.get_user(user_id)
                post_author = network.get_user(post.author_id)

                # 核心：一次LLM调用
                response = simulate_user_response(user, post, post_author)

                # 更新用户记忆
                update_user_memory(user, response)

                # 记录响应
                all_responses.append({
                    "user_id": user_id,
                    "user_persona": user.persona_type,
                    "post_id": post.post_id,
                    "step": step,
                    "response": response
                })

                # 如果产生了新内容，加入下一轮传播
                IF response["action"] IN ["forward", "forward_comment", "create"]:

                    new_content = response["content"] if response["content"] else post.content

                    new_post = Post(
                        post_id = generate_id(),
                        author_id = user_id,
                        content = new_content,
                        post_type = response["action"],
                        original_post_id = post.post_id,
                        timestamp = step + 1
                    )

                    next_wave.append(new_post)
                    all_posts.append(new_post)

        current_wave = next_wave

        IF len(current_wave) == 0:
            BREAK

    RETURN SimulationResult(
        posts = all_posts,
        responses = all_responses
    )
```

### 4.9 生成风险报告

```
FUNCTION generate_risk_report(brand_context, test_content, simulation_result) -> Report:

    # 统计基础数据
    total_reach = len(simulation_result.responses)
    action_counts = count_by_field(simulation_result.responses, "action")
    emotion_counts = count_by_field(simulation_result.responses, "emotion")
    negative_contents = filter_negative(simulation_result.posts)

    prompt = f"""
    【品牌背景】
    {brand_context.description}

    【测试内容】
    {test_content}

    【模拟结果统计】
    - 总触达人数：{total_reach}
    - 行为分布：{action_counts}
    - 情绪分布：{emotion_counts}

    【负面反应样本】
    {format_samples(negative_contents[:10])}

    【各人群典型反应】
    {format_by_persona(simulation_result)}

    ---

    请基于以上模拟结果，生成舆情风险评估报告：

    1. **风险评分**：0-100分，并说明理由

    2. **主要风险点**：列出2-4个，每个说明：
       - 风险描述
       - 触发原因
       - 影响人群

    3. **正面因素**：列出正面反应（如有）

    4. **优化建议**：3-5条具体建议，包含问题、建议、修改示例

    输出JSON格式。
    """

    report = LLM.call(prompt)
    RETURN parse_json(report)
```

---

## 五、算力分析与优化

### 5.1 LLM调用量估算

系统中的LLM调用发生在以下环节：

| 环节         | 调用次数 | 说明             |
| ------------ | -------- | ---------------- |
| 生成人群类型 | 1次      | 一次生成所有人群 |
| 生成用户画像 | N次      | N = 用户数量     |
| 初始化记忆   | N次      | N = 用户数量     |
| 传播模拟     | M次      | M = 总触达次数   |
| 生成报告     | 1次      | 汇总分析         |

**总调用量** ≈ 2N + M + 2

### 5.2 场景估算

假设：

- 用户数量 N = 100
- 模拟步数 = 10
- 平均每步传播到30人
- 总触达 M ≈ 300

| 环节         | 调用次数 | Token估算（输入+输出） |
| ------------ | -------- | ---------------------- |
| 生成人群类型 | 1        | ~2000                  |
| 生成用户画像 | 100      | ~800 × 100 = 80,000    |
| 初始化记忆   | 100      | ~500 × 100 = 50,000    |
| 传播模拟     | 300      | ~1000 × 300 = 300,000  |
| 生成报告     | 1        | ~3000                  |
| **总计**     | **502**  | **~435,000 tokens**    |

### 5.3 成本估算

以主流模型价格估算（2026年1月）：

| 模型          | 输入价格 | 输出价格 | 本场景成本 |
| ------------- | -------- | -------- | ---------- |
| GPT-4o        | $2.5/1M  | $10/1M   | ~$2-3      |
| GPT-4o-mini   | $0.15/1M | $0.6/1M  | ~$0.15-0.2 |
| Claude Sonnet | $3/1M    | $15/1M   | ~$3-5      |
| Claude Haiku  | $0.25/1M | $1.25/1M | ~$0.2-0.3  |

**结论**：100用户规模的单次模拟，成本在 **$0.2 - $5** 之间，取决于模型选择。

## 六、完整调用示例

```python
# 1. 配置品牌
brand_context = BrandContext(
    brand_name = "XX饮料",
    description = "...",
    target_audience = "...",
    sensitivity_topics = ["食品安全", "添加剂", "价格", "虚假宣传"]
)

# 2. 待测内容
test_content = "【重磅升级】全新XX饮料，0糖0卡，健康新选择！现在购买享8折优惠！"

# 3. 构建虚拟世界
persona_types = generate_persona_types(brand_context)       # 1次LLM
graph = build_network_topology(user_count=100)              # 算法
influence_levels = assign_influence_levels(graph)           # 规则
users = generate_users(graph, influence_levels, persona_types)  # 100次LLM（可批量优化）
network = SocialNetwork(graph, users)
initialize_memories(users, brand_context)                   # 100次LLM（可批量优化）

# 4. 运行模拟
result = run_simulation(network, test_content, max_steps=10)  # ~300次LLM

# 5. 生成报告
report = generate_risk_report(brand_context, test_content, result)  # 1次LLM

print(report)
```

---

## 七、输出报告示例

```json
{
  "risk_score": 62,
  "risk_level": "中风险",
  "summary": "内容中'0糖0卡'表述可能触发代糖争议联想，'8折优惠'可能引发'先涨后降'质疑",

  "risk_points": [
    {
      "risk": "代糖争议联想",
      "trigger": "'0糖0卡'表述触发用户对去年代糖事件的记忆",
      "affected_personas": ["成分党", "潜在批评者"],
      "sample_reaction": "每次看到0糖0卡就想问：到底用的什么甜味剂？"
    },
    {
      "risk": "价格策略质疑",
      "trigger": "'8折优惠'让价格敏感用户联想到之前涨价",
      "affected_personas": ["价格敏感者"],
      "sample_reaction": "8折？先涨价再打折的老套路了吧"
    }
  ],

  "positive_factors": [
    "品牌忠粉群体反应积极，愿意转发支持",
    "健康概念本身符合目标受众需求"
  ],

  "recommendations": [
    {
      "issue": "'0糖0卡'表述过于简单，容易引发成分质疑",
      "suggestion": "明确标注使用的甜味剂类型，增加透明度",
      "example": "改为：'采用赤藓糖醇，0糖0卡，安心畅饮'"
    },
    {
      "issue": "'8折优惠'容易引发'先涨后降'联想",
      "suggestion": "改变促销表述方式",
      "example": "改为：'新品尝鲜价，限时特惠'"
    },
    {
      "issue": "缺乏可信背书",
      "suggestion": "添加第三方认证信息",
      "example": "可加入：'通过XX机构检测认证'"
    }
  ]
}
```

---

## 八、设计总结

### 8.1 什么交给LLM

| 任务                       | 方式        |
| -------------------------- | ----------- |
| 人群分类                   | LLM生成     |
| 用户画像                   | LLM生成     |
| 初始记忆                   | LLM生成     |
| 用户反应（情绪+行动+内容） | LLM一次调用 |
| 风险报告                   | LLM生成     |

### 8.2 什么用算法/规则

| 任务           | 方式                 |
| -------------- | -------------------- |
| 网络拓扑构建   | BA无标度网络算法     |
| 影响力角色分配 | 基于入度的百分位规则 |
| 传播队列管理   | 列表操作             |
| 记忆长度限制   | 简单截断             |
| 统计汇总       | 计数和分组           |

### 8.3 不需要的东西

- ❌ 影响力计算公式
- ❌ 态度学习率
- ❌ 情绪阈值
- ❌ 负面放大系数
- ❌ 记忆重要性排序算法
- ❌ 任何数值化的用户心理特征

---

## 九、扩展方向

1. **多轮测试**：对同一内容的多个版本进行对比模拟
2. **历史回测**：用真实舆情事件验证模拟准确性
3. **多平台适配**：通过调整人群描述适配不同平台特性（微博/小红书/抖音）
4. **实时联动**：与真实舆情监测系统对接，持续校准
5. **竞品分析**：模拟竞品内容在同一人群中的表现对比

---

**文档结束**
