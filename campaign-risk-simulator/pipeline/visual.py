import base64
import httpx
from chains.image_chain import describe_image


async def process_visual(image_url: str, campaign_description: str) -> str:
    """从七牛云 URL 下载图片，转 base64，调用视觉 LLM"""
    async with httpx.AsyncClient() as client:
        resp = await client.get(image_url, timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "image/jpeg")
        mime_type = content_type.split(";")[0].strip()
        b64 = base64.b64encode(resp.content).decode()

    return await describe_image(b64, mime_type, campaign_description)
