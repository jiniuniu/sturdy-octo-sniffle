# utils/qiniu_uploader.py
"""
七牛云图片上传工具。

仅在 settings.qiniu_configured == True 时实际上传；
否则返回空字符串，由调用方决定是否跳过。
"""
from config import settings


def upload_bytes(data: bytes, key: str) -> str:
    """
    将 bytes 上传到七牛云，返回公开访问 URL。
    key 建议格式："{job_id}/{chart_name}.png"
    """
    from qiniu import Auth, put_data  # 懒加载，未安装时不影响其他模块

    q = Auth(settings.QINIU_ACCESS_KEY, settings.QINIU_SECRET_KEY)
    token = q.upload_token(settings.QINIU_BUCKET, key)
    ret, info = put_data(token, key, data)
    if info.status_code != 200:
        raise RuntimeError(f"七牛云上传失败 [{key}]: {info}")
    return f"{settings.QINIU_DOMAIN.rstrip('/')}/{ret['key']}"


def upload_charts(matrix, report, job_id: str) -> dict[str, str]:
    """
    生成四张报告图表并上传到七牛云，返回 {chart_name: url}。

    若七牛云未配置，返回空 dict。
    """
    if not settings.qiniu_configured:
        return {}

    from utils.export import (
        build_stats,
        chart_axis_breakdown,
        chart_correlations,
        chart_distribution,
        chart_tpb_barriers,
    )

    stats = build_stats(matrix)
    chart_fns = {
        "distribution": chart_distribution,
        "correlations": chart_correlations,
        "barriers": chart_tpb_barriers,
        "breakdown": chart_axis_breakdown,
    }

    urls: dict[str, str] = {}
    for name, fn in chart_fns.items():
        png_bytes: bytes = fn(stats, save_file=False)
        key = f"{job_id}/{name}.png"
        urls[name] = upload_bytes(png_bytes, key)

    return urls
