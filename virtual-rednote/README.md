# Virtual RedNote

AI 驱动的内容传播仿真平台。品牌输入一段内容，系统在一个虚拟社交社区中模拟它的传播过程，实时可视化触达范围、转发路径和互动指标。

## 工作原理

```
1. 创建社区  →  python create_world.py  生成虚拟用户群体并持久化
2. 输入内容  →  LLM 将文案投射为维度向量
3. 启动模拟  →  规则驱动的虚拟用户在社交网络中传播内容
4. 实时可视化 → D3 传播图 + 评论流 + 指标折线图（SSE 推流）
```

---

## 本地开发

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`：

```ini
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=virtual_rednote

LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o
```

### 3. 创建社区

```bash
python create_world.py --desc "你的品牌/场景描述" --n-agents 100
```

### 4. 启动服务

```bash
uvicorn api.main:app --reload --port 8000
```

打开浏览器访问 `http://localhost:8000`。

---

## 服务器部署（Docker）

> 前提：服务器上已运行 `dbs_backend` docker network（见 `dbs/docker-compose.yml`），MongoDB 容器名为 `mongodb`。

### 部署流程

```bash
# 1. 登录服务器，拉取代码
git pull

# 2. 首次部署：配置环境变量
cp .env.example .env
vi .env  # 填入实际值，注意 MONGODB_URI 用容器名
```

`.env` 示例：

```ini
MONGODB_URI=mongodb://mongodb:27017
MONGODB_DB=virtual_rednote

LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o
```

```bash
# 3. 一键部署（构建镜像 + 启动容器）
./deploy.sh
```

后续更新只需 `git pull && ./deploy.sh`。

### 创建社区

服务启动后，在服务器上执行：

```bash
docker exec -it virtual-rednote python create_world.py \
    --desc "你的品牌/场景描述" \
    --n-agents 100
```

---

## 项目结构

```
virtual-rednote/
├── api/
│   └── main.py          # FastAPI 接口 + SSE 流
├── sim/
│   ├── models.py        # 数据结构
│   ├── world.py         # Agent 生成 + 社交图（Sobol 采样）
│   ├── rules.py         # 行为概率计算
│   ├── engine.py        # 事件驱动仿真引擎
│   └── db.py            # MongoDB 读写封装
├── llm/
│   ├── client.py        # LLM 客户端
│   ├── world_gen.py     # 生成社区维度 + 品牌人设
│   ├── persona.py       # 批量生成 agent persona
│   ├── content.py       # 内容向量化
│   ├── comment.py       # LLM 生成评论
│   └── brand.py         # 品牌 agent 巡查回复
├── web/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── create_world.py      # 离线创建社区脚本
├── config.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── deploy.sh
```

---

## 仿真机制

### 用户分层

| 类型   | 占比 | 特征 |
|--------|------|------|
| KOL    | 5%   | 维度偏好强烈，转发后触发病毒扩散 |
| KOC    | 15%  | 有一定偏好，关注所有 KOL |
| 普通   | 80%  | 偏好均匀分布 |

### 行为决策

```
p_like    = match × activity × 0.6
p_comment = match × expressiveness × 0.3
p_repost  = match × sharing × (1 + social_pressure) × 0.2
```

- `match`：agent 偏好向量与内容向量的 cosine similarity
- `social_pressure`：关注列表中已转发比例（KOL 转发权重 ×3）

### 品牌 Agent

每 30 个仿真事件，品牌方 LLM 巡查新增评论，选择性回复最多 3 条（优先 KOL/KOC 和有购买意向的评论，不重复回复同一用户）。
