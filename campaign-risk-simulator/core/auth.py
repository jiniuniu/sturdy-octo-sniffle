from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config import settings

bearer = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(bearer),
):
    # 未配置 api_keys 时跳过认证（本地开发）
    if not settings.api_keys:
        return None

    if credentials is None or credentials.credentials not in settings.api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return credentials.credentials
