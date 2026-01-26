"""
SentiSim 完整模拟运行脚本

运行方式:
    export OPENROUTER_API_KEY="your-api-key"
    cd sentisim

    # 创建新世界并运行模拟
    python main.py

    # 加载已有世界并运行新模拟
    python main.py --load data/worlds/XX饮料_30users_20260126_120000

    # 只创建世界不运行模拟
    python main.py --create-only

    # 列出已有世界
    python main.py --list-worlds
"""

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger
from sentisim.llm import SentiSimLLM
from sentisim.models import BrandContext
from sentisim.world import World

# ============================================================
# 默认配置
# ============================================================

# 模拟参数
USER_COUNT = 30
MAX_STEPS = 5
PERSONA_COUNT = 6
MAX_CONCURRENT = 5

# 品牌配置
DEFAULT_BRAND = BrandContext(
    brand_name="XX饮料",
    description="""
        XX饮料是一家主打健康概念的饮品品牌，目标客群为18-35岁注重生活品质的年轻人。
        品牌调性偏年轻、活力、健康。
        历史上曾在2024年因代糖成分问题引发争议，后官方澄清但部分消费者仍有疑虑。
        去年有过一次涨价，引发部分老用户不满。
    """,
    target_audience="""
        主要是一二线城市的年轻白领和大学生，关注健康、热爱社交媒体，
        对食品安全和成分比较敏感，价格敏感度中等，容易被KOL种草也容易被负面新闻劝退。
    """,
    sensitivity_topics=["食品安全", "添加剂/代糖", "价格", "虚假宣传"],
)

# 待测内容
DEFAULT_TEST_CONTENT = "【重磅升级】全新XX饮料，0糖0卡，健康新选择！现在购买享8折优惠！"


# ============================================================
# 日志配置
# ============================================================


def setup_logging(verbose: bool = False):
    """配置日志"""
    logger.remove()

    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level=level,
    )


# ============================================================
# 主命令
# ============================================================


async def create_world(
    brand_context: BrandContext = DEFAULT_BRAND,
    user_count: int = USER_COUNT,
    persona_count: int = PERSONA_COUNT,
    max_concurrent: int = MAX_CONCURRENT,
) -> World:
    """创建新世界"""
    llm = SentiSimLLM()

    world = await World.create(
        brand_context=brand_context,
        user_count=user_count,
        persona_count=persona_count,
        max_concurrent=max_concurrent,
        llm=llm,
    )

    # 保存
    world.save()

    logger.info(f"LLM 总调用次数: {llm.call_count}")

    return world


async def run_simulation(
    world: World,
    test_content: str = DEFAULT_TEST_CONTENT,
    max_steps: int = MAX_STEPS,
    max_concurrent: int = MAX_CONCURRENT,
):
    """运行模拟并打印结果"""
    llm = SentiSimLLM()

    def on_step(step, wave_size, new_posts):
        logger.info(
            f"  Step {step}: 处理 {wave_size} 条反应, 产生 {new_posts} 条新帖子"
        )

    result = await world.simulate(
        test_content=test_content,
        max_steps=max_steps,
        max_concurrent=max_concurrent,
        llm=llm,
        on_step_complete=on_step,
    )

    # 打印结果
    print_simulation_result(world, result)

    logger.info(f"LLM 总调用次数: {llm.call_count}")

    return result


def print_simulation_result(world: World, result):
    """打印模拟结果"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("模拟结果")
    logger.info("=" * 60)

    logger.info(f"总触达: {result.total_reach}")
    logger.info(f"运行步数: {result.steps_run}")
    logger.info(f"总帖子: {len(result.posts)}")
    logger.info(f"传播帖子: {len(result.get_spreading_posts())}")

    # 行动分布
    logger.info("")
    logger.info("行动分布:")
    action_counts = result.get_action_counts()
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / result.total_reach * 100 if result.total_reach else 0
        bar = "█" * int(pct / 5)
        logger.info(f"  {action:15} {count:4} ({pct:5.1f}%) {bar}")

    # 情绪分布
    logger.info("")
    logger.info("情绪分布 (Top 10):")
    emotion_counts = result.get_emotion_counts()
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1])[:10]:
        pct = count / result.total_reach * 100 if result.total_reach else 0
        logger.info(f"  {emotion:15} {count:4} ({pct:5.1f}%)")

    # 人群反应
    logger.info("")
    logger.info("人群反应:")
    by_persona = result.get_responses_by_persona()
    for persona, records in sorted(by_persona.items(), key=lambda x: -len(x[1])):
        negative = sum(1 for r in records if r.response.is_negative)
        spread = sum(1 for r in records if r.response.will_spread)
        logger.info(
            f"  {persona:15} 总:{len(records):3} 负面:{negative:2} 传播:{spread:2}"
        )

    # 负面反应样本
    negative_responses = result.get_negative_responses()
    if negative_responses:
        logger.info("")
        logger.info(f"负面反应样本 ({len(negative_responses)} 条):")
        for record in negative_responses[:5]:
            logger.info(f"  [{record.user_persona}] {record.response.first_reaction}")

    # 传播内容样本
    spreading_posts = result.get_spreading_posts()
    if spreading_posts:
        logger.info("")
        logger.info(f"传播内容样本 ({len(spreading_posts)} 条):")
        for post in spreading_posts[:5]:
            author = world.get_user(post.author_id)
            persona = author.persona_type if author else "unknown"
            logger.info(f"  [{post.post_type.value}] by {persona}")
            logger.info(f"    {post.content[:60]}...")


def list_worlds(base_path: str = "data/worlds"):
    """列出所有已保存的世界"""
    worlds_dir = Path(base_path)

    if not worlds_dir.exists():
        logger.info("没有找到已保存的世界")
        return

    logger.info("已保存的世界:")
    logger.info("")

    for world_dir in sorted(worlds_dir.iterdir()):
        if world_dir.is_dir() and (world_dir / "meta.json").exists():
            import json

            with open(world_dir / "meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)

            sim_count = (
                len(list((world_dir / "simulations").glob("sim_*")))
                if (world_dir / "simulations").exists()
                else 0
            )

            logger.info(f"  {meta['world_id']}")
            logger.info(
                f"    用户数: {meta['user_count']}, 人群类型: {meta['persona_count']}"
            )
            logger.info(f"    创建时间: {meta['created_at']}")
            logger.info(f"    已运行模拟: {sim_count} 次")
            logger.info("")


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description="SentiSim 舆情模拟系统")

    parser.add_argument("--load", type=str, help="加载已有世界的路径")
    parser.add_argument(
        "--create-only", action="store_true", help="只创建世界，不运行模拟"
    )
    parser.add_argument(
        "--list-worlds", action="store_true", help="列出所有已保存的世界"
    )
    parser.add_argument(
        "--content", type=str, default=DEFAULT_TEST_CONTENT, help="待测试的内容"
    )
    parser.add_argument(
        "--users", type=int, default=USER_COUNT, help=f"用户数量 (默认: {USER_COUNT})"
    )
    parser.add_argument(
        "--steps", type=int, default=MAX_STEPS, help=f"最大模拟步数 (默认: {MAX_STEPS})"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=MAX_CONCURRENT,
        help=f"LLM 并发数 (默认: {MAX_CONCURRENT})",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    return parser.parse_args()


async def main():
    args = parse_args()
    setup_logging(verbose=args.verbose)

    logger.info("=" * 60)
    logger.info("SentiSim 舆情模拟系统")
    logger.info("=" * 60)

    # 列出世界
    if args.list_worlds:
        list_worlds()
        return

    # 加载或创建世界
    if args.load:
        logger.info(f"加载世界: {args.load}")
        world = World.load(args.load)
    else:
        logger.info("创建新世界...")
        world = await create_world(
            user_count=args.users,
            max_concurrent=args.concurrent,
        )

    # 是否运行模拟
    if args.create_only:
        logger.info("世界创建完成（--create-only 模式，跳过模拟）")
        return

    # 运行模拟
    await run_simulation(
        world=world,
        test_content=args.content,
        max_steps=args.steps,
        max_concurrent=args.concurrent,
    )

    # 显示历史模拟
    simulations = world.list_simulations()
    if len(simulations) > 1:
        logger.info("")
        logger.info(f"该世界共有 {len(simulations)} 次模拟记录")


if __name__ == "__main__":
    asyncio.run(main())
