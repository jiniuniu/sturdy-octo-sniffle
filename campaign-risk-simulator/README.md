# Campaign Risk Simulator

品牌营销活动风险模拟服务，基于 LLM 多 Persona 仿真生成消费者反应与风险评估报告。

---

## 本地开发

### 1. 配置环境变量

创建 `.env`：

```env
LLM_API_KEY=        # OpenAI-compatible 服务的 API Key
LLM_BASE_URL=       # API 地址，如 https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=          # 模型名，如 qwen3.5-flash

QINIU_ACCESS_KEY=
QINIU_SECRET_KEY=
QINIU_BUCKET_NAME=
QINIU_DOMAIN=       # 如 https://cdn.example.com

MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=campaign_risk
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

### 1. 首次部署

```bash
# 克隆代码
git clone <repo_url>
cd campaign-risk-simulator

# 创建 .env，MONGODB_URI 改为容器名
MONGODB_URI=mongodb://mongodb:27017

# 启动基础服务（mongodb）
cd /path/to/dbs && docker compose up -d

# 启动应用
cd campaign-risk-simulator && docker compose up --build -d
```

### 2. 更新部署

```bash
cd campaign-risk-simulator
./deploy.sh
```

---

## API 使用

```bash
# 创建 campaign（纯文字）
curl -X POST http://localhost:6791/campaigns \
  -F "title=新能源汽车春节活动" \
  -F "description=某国内新能源汽车品牌发布春节营销活动..." \
  -F "target_market=中国大陆" \
  -F "n_personas=8"

# 创建 campaign（带图片）
curl -X POST http://localhost:6791/campaigns \
  -F "title=新能源汽车春节活动" \
  -F "description=..." \
  -F "image=@poster.jpg"

# 触发 pipeline
curl -X POST http://localhost:6791/campaigns/{campaign_id}/run

# 轮询状态（建议间隔 5-10s）
curl http://localhost:6791/campaigns/{campaign_id}/status

# 获取完整结果
curl http://localhost:6791/campaigns/{campaign_id}/result

# 查看 HTML 报告
open http://localhost:6791/campaigns/{campaign_id}/report
```

### Pipeline 状态流转

```
idle → processing_visual（有图片时）→ extracting_dimensions
     → sampling → generating_personas → generating_comments → completed
                                                             → error（可重试）
```

- `error` 状态下再次调用 `/run` 会清理上次数据并重新执行
- `completed` 状态不可重跑，需新建 campaign
