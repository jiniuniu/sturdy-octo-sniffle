# SentiSim 舆情风险推演系统

基于 LLM 的社交网络舆情模拟与风险预测系统。

## 快速开始

### 1. 安装依赖

```bash
cd sentisim
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 OpenRouter API Key
```

### 3. 运行测试

```bash
python tests/test_llm.py
```

## 项目结构

```
sentisim/
├── config/
│   └── settings.py          # 全局配置
├── sentisim/
│   ├── models/              # Pydantic 数据模型
│   │   ├── brand.py         # 品牌上下文
│   │   ├── persona.py       # 人群类型
│   │   ├── user.py          # 用户
│   │   ├── post.py          # 帖子
│   │   └── response.py      # 用户反应
│   └── llm/
│       └── client.py        # LLM 客户端封装
└── tests/
    └── test_llm.py          # LLM 测试
```

## 开发进度

- [x] 阶段1：基础设施
  - [x] 项目骨架
  - [x] Pydantic 数据模型
  - [x] LLM 客户端封装（OpenRouter + LangChain）
- [ ] 阶段2：数据生成
  - [ ] 人群类型生成器
  - [ ] 网络拓扑构建
  - [ ] 用户画像生成器
  - [ ] 记忆初始化器
- [ ] 阶段3：模拟引擎
  - [ ] 用户反应模拟
  - [ ] 记忆更新
  - [ ] 传播循环
- [ ] 阶段4：完整流程
  - [ ] 主流程编排
  - [ ] 风险报告生成
  - [ ] CLI 入口
