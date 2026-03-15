# Consumer Research Framework

通用消费者行为研究框架，基于 LLM 多 Persona 仿真，模拟消费者对任意商业决策场景的反应并生成研究报告。

支持场景包括：营销活动风险评估、新品概念测试、定价测试、广告创意测试、品牌主张测试、公关声明评估等。

---

## 本地开发

### 1. 配置环境变量

创建 `.env`：

```env
LLM_API_KEY=        # OpenAI-compatible 服务的 API Key
LLM_BASE_URL=       # API 地址，如 https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=          # 模型名，如 qwen-plus

QINIU_ACCESS_KEY=
QINIU_SECRET_KEY=
QINIU_BUCKET_NAME=
QINIU_DOMAIN=       # 如 https://cdn.example.com

MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=consumer_research
```

### 2. 启动依赖服务

```bash
cd /path/to/dbs
docker compose up -d
```

### 3. 启动服务

```bash
uvicorn main:app --reload --port 6791
```

| 地址 | 说明 |
|------|------|
| `http://localhost:6791` | API 服务 |
| `http://localhost:6791/docs` | Swagger 文档 |

---

## 云端部署

### 首次部署

```bash
git clone <repo_url>
cd campaign-risk-simulator

# 创建 .env，MONGODB_URI 改为容器名
MONGODB_URI=mongodb://mongodb:27017

# 启动基础服务
cd /path/to/dbs && docker compose up -d

# 启动应用
cd campaign-risk-simulator && docker compose up --build -d
```

### 更新部署

```bash
./deploy.sh
```

---

## API 使用

### 新通用接口 `/studies`

```bash
# 创建研究（风险评估示例）
curl -X POST http://localhost:6791/studies \
  -F "title=新能源汽车春节活动" \
  -F "description=某国内新能源汽车品牌发布春节营销活动..." \
  -F 'study_design={
    "study_type": "risk_assessment",
    "research_objective": "识别该活动可能引发的舆论风险",
    "response_mode": "comment",
    "analysis_framework": "risk"
  }' \
  -F "target_market=中国大陆" \
  -F "n_personas=8"

# 创建研究（概念测试示例）
curl -X POST http://localhost:6791/studies \
  -F "title=订阅制会员方案" \
  -F "description=我们计划推出月费99元的会员服务..." \
  -F 'study_design={
    "study_type": "pricing_test",
    "research_objective": "了解目标用户对订阅制定价的接受程度",
    "response_mode": "purchase_intent",
    "analysis_framework": "acceptance"
  }'

# 轮询状态（建议间隔 5-10s）
curl http://localhost:6791/studies/{study_id}/status

# 获取完整结果
curl http://localhost:6791/studies/{study_id}/result

# 查看 HTML 报告
open http://localhost:6791/studies/{study_id}/report
```

### Pipeline 状态流转

```
idle → processing_visual（有图片时）→ extracting_dimensions
     → sampling → generating_personas → generating_responses → completed
                                                              → error
```

### study_design 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `study_type` | string | 研究类型，见下方枚举 |
| `research_objective` | string | 研究目标自由文本，驱动 LLM 视角 |
| `response_mode` | string | 反应收集方式，见下方枚举 |
| `analysis_framework` | string | 分析框架，决定 summary 结构 |
| `structural_dimensions` | string[] | 必须覆盖的人口变量，默认 `["region", "occupation"]` |
| `preset_dimensions` | object[] | 用户预设研究维度，LLM 会补充剩余 |

**study_type 枚举**

| 值 | 说明 |
|---|---|
| `risk_assessment` | 营销活动风险评估 |
| `concept_test` | 新品/功能概念测试 |
| `pricing_test` | 定价测试 |
| `creative_test` | 广告创意/文案测试 |
| `policy_test` | 规则/权益变更测试 |
| `market_entry` | 市场进入评估 |
| `brand_fit` | 品牌/代言人匹配度 |
| `user_journey` | 用户旅程还原 |
| `competitive_perception` | 竞品对比认知 |

**response_mode 枚举**

| 值 | 说明 |
|---|---|
| `comment` | 社媒评论模拟 |
| `survey` | 结构化问卷回答 |
| `interview` | 深度访谈模拟 |
| `purchase_intent` | 购买决策模拟 |
| `reaction` | 即时情绪反应 |

**analysis_framework 枚举**

| 值 | 说明 |
|---|---|
| `risk` | 风险识别 |
| `acceptance` | 接受度测量 |
| `segmentation` | 人群细分洞察 |
| `decision_path` | 决策路径分析 |
| `fit_assessment` | 匹配度评估 |

---

## 向后兼容

原 `/campaigns` 接口保持不变，现有调用方无需修改。

```bash
curl -X POST http://localhost:6791/campaigns \
  -F "title=活动名称" \
  -F "description=活动描述"
```
