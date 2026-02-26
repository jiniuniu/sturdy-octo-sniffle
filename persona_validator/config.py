# config.py
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):

    # LLM
    LLM_API_KEY: str
    LLM_MODEL: str
    LLM_BASE_URL: str
    LLM_MODEL_FAST: str

    # Pipeline
    N_PERSONAS: int = 16
    LLM_TEMPERATURE: float = 0.7

    # TPB 权重（三项之和应为 1.0）
    TPB_WEIGHT_ATTITUDE: float = 0.4
    TPB_WEIGHT_SUBJECTIVE_NORM: float = 0.3
    TPB_WEIGHT_PERCEIVED_CONTROL: float = 0.3

    # MongoDB
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "persona_validator"

    # 七牛云（未配置时跳过图片上传）
    QINIU_ACCESS_KEY: str = ""
    QINIU_SECRET_KEY: str = ""
    QINIU_BUCKET: str = ""
    QINIU_DOMAIN: str = ""  # e.g. "https://cdn.example.com"

    @property
    def qiniu_configured(self) -> bool:
        return bool(
            self.QINIU_ACCESS_KEY
            and self.QINIU_SECRET_KEY
            and self.QINIU_BUCKET
            and self.QINIU_DOMAIN
        )


settings = Settings()
