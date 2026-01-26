"""
SentiSim 全局配置
"""

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """全局配置，支持环境变量和 .env 文件"""

    # OpenRouter API
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # 默认模型
    default_model: str = "google/gemini-3-flash-preview"

    # 模拟参数
    default_user_count: int = 100
    default_max_steps: int = 10
    max_memory_size: int = 10

    # 并发控制
    max_concurrent_calls: int = 10

    # 成本控制
    budget_limit: float = 10.0  # 美元

    # 网络拓扑参数
    ba_model_m: int = 3  # BA模型中每个新节点连接的边数
    brand_follow_ratio: float = 0.7  # 关注品牌账号的用户比例

    # 影响力分层阈值
    kol_percentile: float = 0.03
    active_percentile: float = 0.15
    normal_percentile: float = 0.60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# 全局单例
settings = Settings()
