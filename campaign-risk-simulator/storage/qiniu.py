import asyncio
import uuid
from pathlib import Path

import qiniu
from config import settings


def _upload_sync(data: bytes, key: str) -> str:
    """同步上传，在线程池里调用"""
    q = qiniu.Auth(settings.qiniu_access_key, settings.qiniu_secret_key)
    token = q.upload_token(settings.qiniu_bucket_name, key)
    ret, info = qiniu.put_data(token, key, data)
    if info.status_code != 200:
        raise RuntimeError(f"七牛云上传失败: {info}")
    return key


async def upload_image(data: bytes, filename: str) -> tuple[str, str]:
    """
    上传图片到七牛云
    返回 (key, url)
    """
    suffix = Path(filename).suffix or ".jpg"
    key = f"{settings.qiniu_image_path.strip('/')}/{uuid.uuid4().hex}{suffix}"

    # qiniu SDK 是同步的，放线程池避免阻塞事件循环
    await asyncio.get_event_loop().run_in_executor(None, _upload_sync, data, key)

    url = f"{settings.qiniu_domain.rstrip('/')}/{key}"
    return key, url
