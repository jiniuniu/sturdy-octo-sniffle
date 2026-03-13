from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "campaign_risk"

    # LLM（OpenAI-compatible）
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    vision_model: str = ""         # 留空时回退到 llm_model

    # 七牛云
    qiniu_access_key: str
    qiniu_secret_key: str
    qiniu_bucket_name: str
    qiniu_domain: str              # e.g. https://cdn.example.com
    qiniu_image_path: str = "campaign_risk_simulation/images"

    # 服务
    base_url: str = "http://localhost:8000"

    @property
    def effective_vision_model(self) -> str:
        return self.vision_model or self.llm_model


settings = Settings()
