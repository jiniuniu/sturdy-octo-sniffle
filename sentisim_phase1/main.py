"""
SentiSim 完整模拟运行脚本 (Rich Console 版本)

运行方式:
    export OPENROUTER_API_KEY="your-api-key"
    cd sentisim_phase1

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
import random
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich import box

from sentisim.llm import SentiSimLLM
from sentisim.models import BrandContext, PostType
from sentisim.world import World

# Rich Console
console = Console()

# ============================================================
# 默认配置
# ============================================================

USER_COUNT = 30
MAX_STEPS = 5
PERSONA_COUNT = 6
MAX_CONCURRENT = 5

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

# 多个测试内容
TEST_CONTENTS = [
    {"name": "新品上市+优惠", "content": "【重磅升级】全新XX饮料，0糖0卡，健康新选择！现在购买享8折优惠！"},
    {"name": "明星代言官宣", "content": "官宣！XX饮料正式签约顶流明星@某某某 为品牌代言人！新广告片即将上线，敬请期待~"},
    {"name": "成分升级公告", "content": "【配方升级公告】应广大消费者要求，XX饮料全线产品已更换为天然甜菊糖，告别人工代糖！"},
    {"name": "涨价通知", "content": "【重要通知】因原材料成本上涨，XX饮料将于下月起调整零售价，涨幅约15%。感谢您的理解与支持。"},
    {"name": "争议回应", "content": "针对近期网络上关于XX饮料的不实言论，我们郑重声明：产品完全符合国家食品安全标准，欢迎监督检验。"},
    {"name": "联名活动", "content": "XX饮料 x 热门IP联名款限量发售！集齐6款瓶身可兑换限定周边，快来打卡集卡吧！"},
    {"name": "用户UGC活动", "content": "晒出你和XX饮料的故事，带话题#XX陪我的日子# 发微博，有机会赢取全年免费畅饮！"},
    {"name": "环保倡议", "content": "XX饮料正式启动「绿色瓶盖」计划！即日起，所有产品包装采用100%可回收材料，为地球减负！"},
]

DEFAULT_TEST_CONTENT = TEST_CONTENTS[0]["content"]


# ============================================================
# 显示函数
# ============================================================


def show_header():
    """显示标题"""
    title = Text()
    title.append("SentiSim", style="bold cyan")
    title.append(" 舆情模拟系统", style="bold white")

    console.print(Panel(
        title,
        subtitle="Sentiment Simulation Engine",
        border_style="cyan",
        padding=(1, 2),
    ))


def show_brand_info(brand: BrandContext):
    """显示品牌信息"""
    table = Table(title="品牌信息", box=box.ROUNDED, border_style="blue")
    table.add_column("属性", style="cyan", width=12)
    table.add_column("内容", style="white")

    table.add_row("品牌名称", brand.brand_name)
    table.add_row("目标受众", brand.target_audience.strip()[:60] + "...")
    table.add_row("敏感话题", ", ".join(brand.sensitivity_topics))

    console.print(table)


def show_test_content_menu():
    """显示测试内容菜单"""
    table = Table(title="可选测试内容", box=box.SIMPLE, border_style="yellow")
    table.add_column("#", style="dim", width=3)
    table.add_column("类型", style="cyan", width=15)
    table.add_column("内容预览", style="white")

    for i, tc in enumerate(TEST_CONTENTS, 1):
        table.add_row(str(i), tc["name"], tc["content"][:45] + "...")

    console.print(table)


def show_world_info(world: World):
    """显示世界信息"""
    table = Table(title="虚拟世界", box=box.ROUNDED, border_style="green")
    table.add_column("属性", style="cyan", width=15)
    table.add_column("值", style="white")

    table.add_row("世界ID", world.meta.world_id)
    table.add_row("用户数量", str(world.meta.user_count))
    table.add_row("人群类型数", str(world.meta.persona_count))
    table.add_row("创建时间", world.meta.created_at)

    console.print(table)

    # 显示人群分布
    if world.personas:
        persona_table = Table(title="人群类型分布", box=box.SIMPLE)
        persona_table.add_column("人群", style="cyan")
        persona_table.add_column("占比", style="yellow")
        persona_table.add_column("描述", style="dim")

        for persona in world.personas:
            desc = persona.description[:40] + "..." if len(persona.description) > 40 else persona.description
            persona_table.add_row(
                persona.name,
                f"{persona.percentage:.0%}",
                desc
            )

        console.print(persona_table)


def show_simulation_progress(step: int, wave_size: int, new_posts: int, total_steps: int):
    """显示模拟进度"""
    console.print(f"  [dim]Step {step}/{total_steps}[/dim] "
                  f"处理 [cyan]{wave_size}[/cyan] 条反应, "
                  f"产生 [green]{new_posts}[/green] 条新帖子")


def show_simulation_result(world: World, result):
    """显示模拟结果"""

    # 概览面板
    overview = Table.grid(padding=(0, 2))
    overview.add_column(justify="center")
    overview.add_column(justify="center")
    overview.add_column(justify="center")
    overview.add_column(justify="center")

    overview.add_row(
        f"[bold cyan]{result.total_reach}[/]\n[dim]总触达[/]",
        f"[bold green]{result.steps_run}[/]\n[dim]运行步数[/]",
        f"[bold yellow]{len(result.posts)}[/]\n[dim]总帖子[/]",
        f"[bold magenta]{len(result.get_spreading_posts())}[/]\n[dim]传播帖子[/]",
    )

    console.print(Panel(overview, title="模拟结果概览", border_style="cyan"))

    # 行动分布表
    action_table = Table(title="行动分布", box=box.ROUNDED)
    action_table.add_column("行动", style="cyan", width=18)
    action_table.add_column("数量", style="white", justify="right", width=6)
    action_table.add_column("占比", style="yellow", justify="right", width=8)
    action_table.add_column("分布", style="green", width=20)

    action_counts = result.get_action_counts()
    max_count = max(action_counts.values()) if action_counts.values() else 1

    action_labels = {
        "ignore": ("划走", "dim"),
        "like": ("点赞", "yellow"),
        "forward": ("转发", "green"),
        "forward_comment": ("转发评论", "cyan"),
        "create": ("二创", "magenta"),
    }

    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / result.total_reach * 100 if result.total_reach else 0
        bar_len = int(count / max_count * 15) if max_count else 0
        label, color = action_labels.get(action, (action, "white"))
        action_table.add_row(
            f"[{color}]{label}[/]",
            str(count),
            f"{pct:.1f}%",
            f"[{color}]{'█' * bar_len}[/]"
        )

    console.print(action_table)

    # 情绪分布表
    emotion_table = Table(title="情绪分布 (Top 8)", box=box.ROUNDED)
    emotion_table.add_column("情绪", style="white", width=20)
    emotion_table.add_column("数量", style="cyan", justify="right", width=6)
    emotion_table.add_column("占比", style="yellow", justify="right", width=8)

    emotion_counts = result.get_emotion_counts()
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1])[:8]:
        pct = count / result.total_reach * 100 if result.total_reach else 0
        # 根据情绪类型着色
        if any(kw in emotion for kw in ["反感", "愤怒", "担忧", "吐槽", "质疑"]):
            style = "red"
        elif any(kw in emotion for kw in ["开心", "喜欢", "期待", "支持"]):
            style = "green"
        else:
            style = "white"
        emotion_table.add_row(f"[{style}]{emotion}[/]", str(count), f"{pct:.1f}%")

    console.print(emotion_table)

    # 人群反应表
    persona_table = Table(title="人群反应分析", box=box.ROUNDED)
    persona_table.add_column("人群", style="cyan", width=18)
    persona_table.add_column("总数", justify="right", width=6)
    persona_table.add_column("负面", style="red", justify="right", width=6)
    persona_table.add_column("传播", style="green", justify="right", width=6)
    persona_table.add_column("负面率", style="yellow", justify="right", width=8)

    by_persona = result.get_responses_by_persona()
    for persona, records in sorted(by_persona.items(), key=lambda x: -len(x[1])):
        total = len(records)
        negative = sum(1 for r in records if r.response.is_negative)
        spread = sum(1 for r in records if r.response.will_spread)
        neg_rate = negative / total * 100 if total else 0
        persona_table.add_row(
            persona,
            str(total),
            str(negative),
            str(spread),
            f"{neg_rate:.0f}%"
        )

    console.print(persona_table)

    # 负面反应样本
    negative_responses = result.get_negative_responses()
    if negative_responses:
        console.print(f"\n[bold red]负面反应样本[/] ({len(negative_responses)} 条)")
        for record in negative_responses[:5]:
            console.print(f"  [dim]>[/] [cyan][{record.user_persona}][/] {record.response.first_reaction}")

    # 传播内容样本
    spreading_posts = result.get_spreading_posts()
    if spreading_posts:
        console.print(f"\n[bold green]传播内容样本[/] ({len(spreading_posts)} 条)")
        for post in spreading_posts[:5]:
            author = world.get_user(post.author_id)
            persona = author.persona_type if author else "unknown"
            type_color = {
                PostType.FORWARD: "green",
                PostType.FORWARD_COMMENT: "cyan",
            }.get(post.post_type, "white")
            console.print(f"  [dim]>[/] [{type_color}][{post.post_type.value}][/] by [cyan]{persona}[/]")
            console.print(f"    [dim]{post.content[:70]}...[/]")


def show_propagation_tree(result, world: World):
    """显示传播树"""
    tree = Tree("[bold cyan]传播链路[/]")

    # 找到品牌原帖
    brand_post = next((p for p in result.posts if p.post_type == PostType.BRAND_ORIGINAL), None)
    if not brand_post:
        return

    brand_node = tree.add(f"[bold yellow]{brand_post.content[:40]}...[/] [dim](品牌原帖)[/]")

    # 构建传播树
    def add_children(parent_node, parent_post_id, depth=0):
        if depth > 3:  # 限制深度
            return
        children = [p for p in result.posts if p.original_post_id == parent_post_id]
        for child in children[:5]:  # 限制每层子节点数
            author = world.get_user(child.author_id)
            persona = author.persona_type if author else "unknown"
            type_color = {
                PostType.FORWARD: "green",
                PostType.FORWARD_COMMENT: "cyan",
            }.get(child.post_type, "white")

            label = f"[{type_color}]{child.post_type.value}[/] by [dim]{persona}[/]"
            child_node = parent_node.add(label)
            add_children(child_node, child.post_id, depth + 1)

    add_children(brand_node, brand_post.post_id)

    console.print(tree)


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

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]创建虚拟世界...", total=100)

        # 模拟进度（实际创建是异步的）
        progress.update(task, advance=10, description="[cyan]构建网络拓扑...")

        world = await World.create(
            brand_context=brand_context,
            user_count=user_count,
            persona_count=persona_count,
            max_concurrent=max_concurrent,
            llm=llm,
        )

        progress.update(task, advance=90, description="[green]世界创建完成!")

    world.save()
    console.print(f"[green]✓[/] 世界已保存，LLM调用: {llm.call_count} 次")

    return world


async def run_simulation(
    world: World,
    test_content: str = DEFAULT_TEST_CONTENT,
    max_steps: int = MAX_STEPS,
    max_concurrent: int = MAX_CONCURRENT,
):
    """运行模拟"""
    llm = SentiSimLLM()

    console.print(Panel(
        f"[white]{test_content}[/]",
        title="[yellow]测试内容[/]",
        border_style="yellow",
    ))

    console.print("\n[bold]开始模拟传播...[/]\n")

    def on_step(step, wave_size, new_posts):
        show_simulation_progress(step, wave_size, new_posts, max_steps)

    result = await world.simulate(
        test_content=test_content,
        max_steps=max_steps,
        max_concurrent=max_concurrent,
        llm=llm,
        on_step_complete=on_step,
    )

    console.print(f"\n[green]✓[/] 模拟完成，LLM调用: {llm.call_count} 次\n")

    # 显示结果
    show_simulation_result(world, result)

    # 显示传播树
    console.print()
    show_propagation_tree(result, world)

    return result


def list_worlds(base_path: str = "data/worlds"):
    """列出所有已保存的世界"""
    worlds_dir = Path(base_path)

    if not worlds_dir.exists():
        console.print("[yellow]没有找到已保存的世界[/]")
        return

    table = Table(title="已保存的世界", box=box.ROUNDED)
    table.add_column("世界ID", style="cyan")
    table.add_column("用户数", justify="right")
    table.add_column("人群类型", justify="right")
    table.add_column("创建时间", style="dim")
    table.add_column("模拟次数", justify="right", style="yellow")

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

            table.add_row(
                meta['world_id'],
                str(meta['user_count']),
                str(meta['persona_count']),
                meta['created_at'],
                str(sim_count),
            )

    console.print(table)


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description="SentiSim 舆情模拟系统")

    parser.add_argument("--load", type=str, help="加载已有世界的路径")
    parser.add_argument("--create-only", action="store_true", help="只创建世界，不运行模拟")
    parser.add_argument("--list-worlds", action="store_true", help="列出所有已保存的世界")
    parser.add_argument("--content", type=str, help="待测试的内容（或使用 --random 随机选择）")
    parser.add_argument("--random", action="store_true", help="随机选择测试内容")
    parser.add_argument("--users", type=int, default=USER_COUNT, help=f"用户数量 (默认: {USER_COUNT})")
    parser.add_argument("--steps", type=int, default=MAX_STEPS, help=f"最大模拟步数 (默认: {MAX_STEPS})")
    parser.add_argument("--concurrent", type=int, default=MAX_CONCURRENT, help=f"LLM 并发数 (默认: {MAX_CONCURRENT})")

    return parser.parse_args()


async def main():
    args = parse_args()

    show_header()

    # 列出世界
    if args.list_worlds:
        list_worlds()
        return

    # 显示品牌信息
    show_brand_info(DEFAULT_BRAND)

    # 加载或创建世界
    if args.load:
        console.print(f"\n[cyan]加载世界:[/] {args.load}")
        world = World.load(args.load)
    else:
        console.print(f"\n[cyan]创建新世界[/] (用户数: {args.users})")
        world = await create_world(
            user_count=args.users,
            max_concurrent=args.concurrent,
        )

    show_world_info(world)

    # 是否运行模拟
    if args.create_only:
        console.print("\n[yellow]世界创建完成（--create-only 模式，跳过模拟）[/]")
        return

    # 确定测试内容
    if args.random:
        test_case = random.choice(TEST_CONTENTS)
        console.print(f"\n[yellow]随机选择测试内容:[/] {test_case['name']}")
        test_content = test_case["content"]
    elif args.content:
        test_content = args.content
    else:
        show_test_content_menu()
        test_content = DEFAULT_TEST_CONTENT

    # 运行模拟
    console.print()
    await run_simulation(
        world=world,
        test_content=test_content,
        max_steps=args.steps,
        max_concurrent=args.concurrent,
    )

    # 显示历史模拟
    simulations = world.list_simulations()
    if len(simulations) > 1:
        console.print(f"\n[dim]该世界共有 {len(simulations)} 次模拟记录[/]")


if __name__ == "__main__":
    asyncio.run(main())
