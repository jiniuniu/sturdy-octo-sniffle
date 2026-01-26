"""
页面2: 创建世界
"""

import asyncio
import sys

import streamlit as st

sys.path.insert(0, ".")
from sentisim.llm import SentiSimLLM
from sentisim.models import BrandContext
from sentisim.world import World

# 默认值
DEFAULT_BRAND_NAME = "XX饮料"
DEFAULT_DESCRIPTION = """XX饮料是一家主打健康概念的饮品品牌，目标客群为18-35岁注重生活品质的年轻人。
品牌调性偏年轻、活力、健康。
历史上曾在2024年因代糖成分问题引发争议，后官方澄清但部分消费者仍有疑虑。
去年有过一次涨价，引发部分老用户不满。"""

DEFAULT_AUDIENCE = """主要是一二线城市的年轻白领和大学生，关注健康、热爱社交媒体，
对食品安全和成分比较敏感，价格敏感度中等，容易被KOL种草也容易被负面新闻劝退。"""

DEFAULT_TOPICS = ["食品安全", "添加剂/代糖", "价格", "虚假宣传"]


def render():
    st.title("⚙️ 创建新世界")
    st.markdown("配置品牌信息和模拟参数，生成虚拟用户群体")

    # 检查是否正在创建
    if st.session_state.get("creating_world"):
        render_progress()
        return

    render_form()


def render_form():
    """渲染配置表单"""

    with st.form("create_world_form"):
        st.subheader("品牌信息")

        brand_name = st.text_input(
            "品牌名称",
            value=DEFAULT_BRAND_NAME,
            help="品牌或产品名称",
        )

        description = st.text_area(
            "品牌描述",
            value=DEFAULT_DESCRIPTION,
            height=150,
            help="包含行业、调性、历史事件等背景信息",
        )

        audience = st.text_area(
            "目标受众",
            value=DEFAULT_AUDIENCE,
            height=100,
            help="描述目标用户群体的特征",
        )

        topics_input = st.text_input(
            "敏感议题",
            value=", ".join(DEFAULT_TOPICS),
            help="用逗号分隔多个议题",
        )

        st.divider()
        st.subheader("模拟参数")

        col1, col2 = st.columns(2)

        with col1:
            user_count = st.number_input(
                "用户数量",
                min_value=10,
                max_value=500,
                value=30,
                step=10,
                help="建议: 30-200，越多越贵",
            )

            persona_count = st.number_input(
                "人群类型数",
                min_value=3,
                max_value=12,
                value=6,
                help="建议: 4-8种",
            )

        with col2:
            max_concurrent = st.number_input(
                "LLM并发数",
                min_value=1,
                max_value=20,
                value=5,
                help="并发调用数，太高可能触发限流",
            )

            model = st.selectbox(
                "模型",
                options=["google/gemini-3-flash-preview"],
                index=0,
            )

        # 预估
        estimated_calls = 1 + user_count * 2  # 人群类型 + 画像 + 记忆
        st.info(f"预估 LLM 调用次数: ~{estimated_calls} 次")

        submitted = st.form_submit_button(
            "🚀 开始创建", use_container_width=True, type="primary"
        )

        if submitted:
            # 解析敏感议题
            topics = [t.strip() for t in topics_input.split(",") if t.strip()]

            # 保存配置到 session
            st.session_state.create_config = {
                "brand_context": BrandContext(
                    brand_name=brand_name,
                    description=description,
                    target_audience=audience,
                    sensitivity_topics=topics,
                ),
                "user_count": user_count,
                "persona_count": persona_count,
                "max_concurrent": max_concurrent,
                "model": model,
            }
            st.session_state.creating_world = True
            st.session_state.create_progress = []
            st.rerun()


def render_progress():
    """渲染创建进度"""

    st.subheader("创建进度")

    config = st.session_state.create_config

    # 显示配置摘要
    st.markdown(
        f"""
    - **品牌:** {config['brand_context'].brand_name}
    - **用户数:** {config['user_count']}
    - **人群类型:** {config['persona_count']}
    - **模型:** {config['model']}
    """
    )

    st.divider()

    # 进度容器
    progress_container = st.container()
    status_text = st.empty()
    progress_bar = st.progress(0)

    # 运行创建
    async def create():
        progress_steps = [
            "构建网络拓扑",
            "分配影响力角色",
            "生成人群类型",
            "生成用户画像",
            "初始化记忆",
        ]

        try:
            llm = SentiSimLLM(model=config["model"])

            # 更新进度的回调
            def update_progress(step_idx, message=""):
                pct = (step_idx + 1) / len(progress_steps)
                progress_bar.progress(pct)
                status_text.markdown(f"**{progress_steps[step_idx]}** {message}")

            update_progress(0)

            world = await World.create(
                brand_context=config["brand_context"],
                user_count=config["user_count"],
                persona_count=config["persona_count"],
                max_concurrent=config["max_concurrent"],
                llm=llm,
            )

            # 保存
            world.save()

            # 更新 session
            st.session_state.current_world = world
            st.session_state.creating_world = False

            progress_bar.progress(1.0)
            status_text.markdown("**✅ 创建完成!**")

            return world, llm.call_count

        except Exception as e:
            st.session_state.creating_world = False
            raise e

    # 执行
    try:
        world, call_count = asyncio.run(create())

        st.success(f"世界创建成功！LLM 调用次数: {call_count}")
        st.balloons()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 去运行模拟", use_container_width=True, type="primary"):
                st.session_state.creating_world = False
                st.rerun()
        with col2:
            if st.button("➕ 创建另一个", use_container_width=True):
                st.session_state.creating_world = False
                st.session_state.create_config = None
                st.rerun()

    except Exception as e:
        st.error(f"创建失败: {e}")
        if st.button("重试"):
            st.session_state.creating_world = False
            st.rerun()
