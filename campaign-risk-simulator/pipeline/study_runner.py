import asyncio
import logging
import time
from datetime import datetime

import httpx

from db.repositories import dimension_repo, persona_repo, response_repo, study_repo
from models.study import StudyStatus, StudyDesign, ResponseMode, AnalysisFramework
from pipeline.visual import process_visual
from pipeline.sampling import sample_persona_positions, format_segments_text
from chains.dimension_chain import extract_dimensions
from chains.persona_chain import generate_persona
from chains.response_chain import generate_response
from chains.summary_chain import generate_study_summary

logger = logging.getLogger(__name__)


async def _generate_and_save_persona(
    study_id: str,
    stimulus_description: str,
    target_market: str,
    dimensions: list[dict],
    position: dict,
    study_design: StudyDesign,
):
    pid = position["id"]
    t0 = time.monotonic()
    logger.info("[persona] 开始生成: %s", pid)
    try:
        segments_text = format_segments_text(position, dimensions)
        output = await generate_persona(
            campaign_description=stimulus_description,
            segments_text=segments_text,
            persona_id=pid,
            target_market=target_market,
        )
        await persona_repo.update_for_study(study_id, pid, output)
        logger.info("[persona] 完成: %s，耗时 %.1fs", pid, time.monotonic() - t0)
    except Exception:
        logger.exception("[persona] 生成失败: %s，耗时 %.1fs", pid, time.monotonic() - t0)
        raise


async def _generate_and_save_response(
    study_id: str,
    stimulus_description: str,
    persona: dict,
    response_mode: ResponseMode,
    research_objective: str,
):
    pid = persona.get("persona_id")
    t0 = time.monotonic()
    logger.info("[response] 开始生成: %s", pid)
    try:
        output = await generate_response(
            stimulus_description=stimulus_description,
            persona=persona,
            response_mode=response_mode,
            research_objective=research_objective,
        )
        await response_repo.insert(study_id, output, response_mode)
        logger.info("[response] 完成: %s，耗时 %.1fs", pid, time.monotonic() - t0)
    except Exception:
        logger.exception("[response] 生成失败: %s，耗时 %.1fs", pid, time.monotonic() - t0)
        raise


async def _fire_webhook(url: str, payload: dict):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json=payload)
        logger.info("[webhook] 回调成功: %s", url)
    except Exception:
        logger.warning("[webhook] 回调失败: %s", url)


async def run_study_pipeline(study_id: str):
    t_start = time.monotonic()
    logger.info("===== Study Pipeline 启动: study_id=%s =====", study_id)

    try:
        study = await study_repo.get(study_id)
        stimulus_description = study["description"]
        study_design_raw = study["study_design"]
        study_design = StudyDesign(**study_design_raw)
        response_mode = study_design.response_mode
        research_objective = study_design.research_objective
        analysis_framework = study_design.analysis_framework
        target_market = study.get("target_market", "中国大陆")
        n_personas = study.get("n_personas", 8)

        # Step 1: 图片处理（可选）
        if study.get("image_url"):
            logger.info("[Step 1] 开始处理图片")
            await study_repo.update_status(study_id, StudyStatus.processing_visual)
            t0 = time.monotonic()
            visual_desc = await process_visual(study["image_url"], stimulus_description)
            await study_repo.update_visual(study_id, visual_desc)
            visual_description = visual_desc
            logger.info("[Step 1] 图片处理完成，耗时 %.1fs", time.monotonic() - t0)
        else:
            visual_description = ""
            logger.info("[Step 1] 无图片，跳过")

        # Step 2: 提取研究维度
        logger.info("[Step 2] 开始提取研究维度")
        await study_repo.update_status(study_id, StudyStatus.extracting_dimensions)
        t0 = time.monotonic()
        dimensions_output = await extract_dimensions(
            stimulus_description=stimulus_description,
            visual_description=visual_description,
            target_market=target_market,
            study_design=study_design,
        )
        await dimension_repo.insert_many_for_study(study_id, dimensions_output.dimensions)
        dimensions = await dimension_repo.get_by_study(study_id)
        logger.info(
            "[Step 2] 维度提取完成，共 %d 个维度，耗时 %.1fs",
            len(dimensions), time.monotonic() - t0,
        )

        # Step 3: Sobol 采样
        logger.info("[Step 3] 开始 Sobol 采样")
        await study_repo.update_status(study_id, StudyStatus.sampling)
        positions = sample_persona_positions(dimensions, n_personas)
        await persona_repo.insert_placeholders_for_study(study_id, positions)
        logger.info("[Step 3] 采样完成，生成 %d 个 persona 占位", n_personas)

        # Step 4: 并行生成 Persona
        logger.info("[Step 4] 开始并行生成 %d 个 persona", n_personas)
        await study_repo.update_status(study_id, StudyStatus.generating_personas)
        t0 = time.monotonic()
        results = await asyncio.gather(
            *[
                _generate_and_save_persona(
                    study_id, stimulus_description, target_market,
                    dimensions, pos, study_design,
                )
                for pos in positions
            ],
            return_exceptions=True,
        )
        failed = [r for r in results if isinstance(r, Exception)]
        logger.info(
            "[Step 4] persona 生成完成，成功 %d / 失败 %d，耗时 %.1fs",
            len(results) - len(failed), len(failed), time.monotonic() - t0,
        )

        # Step 5: 并行生成反应
        personas = await persona_repo.get_by_study(study_id)
        completed_personas = [p for p in personas if p.get("completed")]
        logger.info("[Step 5] 开始并行生成 %d 条反应（mode=%s）", len(completed_personas), response_mode)
        await study_repo.update_status(study_id, StudyStatus.generating_responses)
        t0 = time.monotonic()
        results = await asyncio.gather(
            *[
                _generate_and_save_response(
                    study_id, stimulus_description, persona,
                    response_mode, research_objective,
                )
                for persona in completed_personas
            ],
            return_exceptions=True,
        )
        failed = [r for r in results if isinstance(r, Exception)]
        logger.info(
            "[Step 5] 反应生成完成，成功 %d / 失败 %d，耗时 %.1fs",
            len(results) - len(failed), len(failed), time.monotonic() - t0,
        )

        # Step 6: 生成研究总结
        logger.info("[Step 6] 开始生成研究总结（framework=%s）", analysis_framework)
        t0 = time.monotonic()
        all_responses = await response_repo.get_by_study(study_id)
        personas_map = {p["persona_id"]: p for p in personas}
        summary = await generate_study_summary(
            stimulus_description=stimulus_description,
            dimensions=dimensions,
            responses=all_responses,
            personas_map=personas_map,
            analysis_framework=analysis_framework,
            research_objective=research_objective,
        )
        await study_repo.update_summary(study_id, summary.model_dump())
        logger.info("[Step 6] 研究总结生成完成，耗时 %.1fs", time.monotonic() - t0)

        await study_repo.update_status(study_id, StudyStatus.completed)
        logger.info(
            "===== Study Pipeline 完成: study_id=%s，总耗时 %.1fs =====",
            study_id, time.monotonic() - t_start,
        )

        callback_url = study.get("callback_url")
        if callback_url:
            await _fire_webhook(callback_url, {"study_id": study_id, "status": "completed"})

    except Exception as e:
        logger.exception("===== Study Pipeline 异常终止: study_id=%s =====", study_id)
        await study_repo.set_error(study_id, str(e))
        study = await study_repo.get(study_id)
        callback_url = study.get("callback_url") if study else None
        if callback_url:
            await _fire_webhook(callback_url, {"study_id": study_id, "status": "error", "error": str(e)})
