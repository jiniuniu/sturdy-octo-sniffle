import asyncio
import logging
import time
from datetime import datetime

from db.repositories import campaign_repo, dimension_repo, persona_repo, comment_repo
from models.campaign import CampaignStatus
from pipeline.visual import process_visual
from pipeline.sampling import sample_persona_positions, format_segments_text
from chains.dimension_chain import extract_dimensions
from chains.persona_chain import generate_persona
from chains.comment_chain import generate_comment
from chains.summary_chain import generate_summary

logger = logging.getLogger(__name__)


async def _generate_and_save_persona(
    campaign_id: str,
    campaign_description: str,
    target_market: str,
    dimensions: list[dict],
    position: dict,
):
    pid = position["id"]
    t0 = time.monotonic()
    logger.info("[persona] 开始生成: %s", pid)
    try:
        segments_text = format_segments_text(position, dimensions)
        output = await generate_persona(
            campaign_description=campaign_description,
            segments_text=segments_text,
            persona_id=pid,
            target_market=target_market,
        )
        await persona_repo.update(campaign_id, pid, output)
        logger.info("[persona] 完成: %s，耗时 %.1fs", pid, time.monotonic() - t0)
    except Exception:
        logger.exception("[persona] 生成失败: %s，耗时 %.1fs", pid, time.monotonic() - t0)
        raise


async def _generate_and_save_comment(
    campaign_id: str,
    campaign_description: str,
    persona: dict,
):
    pid = persona.get("persona_id")
    t0 = time.monotonic()
    logger.info("[comment] 开始生成: %s", pid)
    try:
        output = await generate_comment(
            campaign_description=campaign_description,
            persona=persona,
        )
        await comment_repo.insert(campaign_id, output)
        logger.info("[comment] 完成: %s，耗时 %.1fs", pid, time.monotonic() - t0)
    except Exception:
        logger.exception("[comment] 生成失败: %s，耗时 %.1fs", pid, time.monotonic() - t0)
        raise


async def run_pipeline(campaign_id: str):
    t_start = time.monotonic()
    logger.info("===== Pipeline 启动: campaign_id=%s =====", campaign_id)

    try:
        campaign = await campaign_repo.get(campaign_id)
        campaign_description = campaign["description"]

        # Step 1: 图片处理（可选）
        if campaign.get("image_url"):
            logger.info("[Step 1] 开始处理图片")
            await campaign_repo.update_status(campaign_id, CampaignStatus.processing_visual)
            t0 = time.monotonic()
            visual_desc = await process_visual(campaign["image_url"], campaign_description)
            await campaign_repo.update_visual(campaign_id, visual_desc)
            visual_description = visual_desc
            logger.info("[Step 1] 图片处理完成，耗时 %.1fs", time.monotonic() - t0)
        else:
            visual_description = ""
            logger.info("[Step 1] 无图片，跳过")

        # Step 2: 提取风险维度
        logger.info("[Step 2] 开始提取风险维度")
        await campaign_repo.update_status(campaign_id, CampaignStatus.extracting_dimensions)
        t0 = time.monotonic()
        target_market = campaign.get("target_market", "中国大陆")
        dimensions_output = await extract_dimensions(campaign_description, visual_description, target_market)
        await dimension_repo.insert_many(campaign_id, dimensions_output.dimensions)
        dimensions = await dimension_repo.get_by_campaign(campaign_id)
        logger.info(
            "[Step 2] 维度提取完成，共 %d 个维度，耗时 %.1fs",
            len(dimensions), time.monotonic() - t0,
        )
        for dim in dimensions:
            logger.info("  · %s（%d 个 segment）", dim["name"], len(dim["segments"]))

        # Step 3: Sobol 采样 + 预写入占位 persona
        logger.info("[Step 3] 开始 Sobol 采样")
        await campaign_repo.update_status(campaign_id, CampaignStatus.sampling)
        n_personas = campaign.get("n_personas", 8)
        positions = sample_persona_positions(dimensions, n_personas)
        await persona_repo.insert_placeholders(campaign_id, positions)
        logger.info("[Step 3] 采样完成，生成 %d 个 persona 占位", n_personas)

        # Step 4: 并行生成 Persona
        logger.info("[Step 4] 开始并行生成 %d 个 persona", n_personas)
        await campaign_repo.update_status(campaign_id, CampaignStatus.generating_personas)
        t0 = time.monotonic()
        results = await asyncio.gather(
            *[
                _generate_and_save_persona(campaign_id, campaign_description, target_market, dimensions, pos)
                for pos in positions
            ],
            return_exceptions=True,
        )
        failed = [r for r in results if isinstance(r, Exception)]
        logger.info(
            "[Step 4] persona 生成完成，成功 %d / 失败 %d，耗时 %.1fs",
            len(results) - len(failed), len(failed), time.monotonic() - t0,
        )
        if failed:
            for e in failed:
                logger.error("[Step 4] 失败详情: %s", e)

        # Step 5: 并行生成评论（只对已完成的 persona 生成）
        personas = await persona_repo.get_by_campaign(campaign_id)
        completed_personas = [p for p in personas if p.get("completed")]
        logger.info("[Step 5] 开始并行生成 %d 条评论", len(completed_personas))
        await campaign_repo.update_status(campaign_id, CampaignStatus.generating_comments)
        t0 = time.monotonic()
        results = await asyncio.gather(
            *[
                _generate_and_save_comment(campaign_id, campaign_description, persona)
                for persona in completed_personas
            ],
            return_exceptions=True,
        )
        failed = [r for r in results if isinstance(r, Exception)]
        logger.info(
            "[Step 5] 评论生成完成，成功 %d / 失败 %d，耗时 %.1fs",
            len(results) - len(failed), len(failed), time.monotonic() - t0,
        )
        if failed:
            for e in failed:
                logger.error("[Step 5] 失败详情: %s", e)

        # Step 6: 生成执行摘要
        logger.info("[Step 6] 开始生成执行摘要")
        t0 = time.monotonic()
        all_comments = await comment_repo.get_by_campaign(campaign_id)
        personas_map = {p["persona_id"]: p for p in personas}
        summary = await generate_summary(
            campaign_description=campaign_description,
            dimensions=dimensions,
            comments=all_comments,
            personas_map=personas_map,
        )
        await campaign_repo.update_summary(campaign_id, summary.model_dump())
        logger.info("[Step 6] 执行摘要生成完成，耗时 %.1fs", time.monotonic() - t0)

        await campaign_repo.update_status(campaign_id, CampaignStatus.completed)
        logger.info(
            "===== Pipeline 完成: campaign_id=%s，总耗时 %.1fs =====",
            campaign_id, time.monotonic() - t_start,
        )

    except Exception as e:
        logger.exception("===== Pipeline 异常终止: campaign_id=%s =====", campaign_id)
        await campaign_repo.set_error(campaign_id, str(e))
