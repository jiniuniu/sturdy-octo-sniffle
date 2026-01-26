"""
页面3: 运行模拟
"""

import asyncio
import json
from pathlib import Path

import streamlit as st
from sentisim.llm import SentiSimLLM

DEFAULT_CONTENT = "【重磅升级】全新XX饮料，0糖0卡，健康新选择！现在购买享8折优惠！"


def get_simulation_list(world) -> list[dict]:
    """获取世界的所有模拟"""
    world_dir = Path("data/worlds") / world.meta.world_id
    sim_dir = world_dir / "simulations"

    if not sim_dir.exists():
        return []

    simulations = []
    for sim_path in sorted(sim_dir.glob("sim_*"), reverse=True):
        config_file = sim_path / "config.json"
        result_file = sim_path / "result.json"

        if config_file.exists() and result_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            with open(result_file, "r", encoding="utf-8") as f:
                result = json.load(f)

            simulations.append(
                {
                    "sim_id": config["sim_id"],
                    "path": str(sim_path),
                    "created_at": config["created_at"],
                    "test_content": config["test_content"],
                    "total_reach": result["total_reach"],
                    "negative_count": result["negative_count"],
                    "negative_rate": (
                        result["negative_count"] / result["total_reach"]
                        if result["total_reach"] > 0
                        else 0
                    ),
                }
            )

    return simulations


def render():
    st.title("🚀 运行模拟")

    # 检查是否有加载的世界
    if not st.session_state.current_world:
        st.warning("请先在「世界管理」中加载一个世界，或在「创建世界」中创建新世界")
        return

    world = st.session_state.current_world

    # 检查是否正在运行
    if st.session_state.get("running_simulation"):
        render_progress(world)
        return

    render_form(world)


def render_form(world):
    """渲染模拟配置表单"""

    # 世界信息
    st.markdown(
        f"""
    **当前世界:** {world.meta.world_id}  
    用户: {world.meta.user_count} | 人群: {world.meta.persona_count}种
    """
    )

    st.divider()

    # 模拟配置
    with st.form("run_simulation_form"):
        st.subheader("待测内容")

        test_content = st.text_area(
            "输入要测试的营销内容/文案",
            value=DEFAULT_CONTENT,
            height=100,
            label_visibility="collapsed",
        )

        st.subheader("模拟参数")

        col1, col2 = st.columns(2)

        with col1:
            max_steps = st.number_input(
                "最大步数",
                min_value=1,
                max_value=20,
                value=5,
                help="传播模拟的最大轮数",
            )

        with col2:
            max_concurrent = st.number_input(
                "LLM并发数",
                min_value=1,
                max_value=20,
                value=5,
            )

        reset_memory = st.checkbox(
            "重置用户记忆",
            value=True,
            help="每次模拟从初始状态开始，不受之前模拟影响",
        )

        submitted = st.form_submit_button(
            "▶️ 开始模拟", use_container_width=True, type="primary"
        )

        if submitted:
            st.session_state.sim_config = {
                "test_content": test_content,
                "max_steps": max_steps,
                "max_concurrent": max_concurrent,
                "reset_memory": reset_memory,
            }
            st.session_state.running_simulation = True
            st.rerun()

    # 历史模拟
    st.divider()
    st.subheader("历史模拟")

    simulations = get_simulation_list(world)

    if not simulations:
        st.info("该世界还没有运行过模拟")
    else:
        for sim in simulations:
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                # 负面率颜色
                neg_rate = sim["negative_rate"]
                if neg_rate < 0.15:
                    color = "🟢"
                elif neg_rate < 0.30:
                    color = "🟡"
                else:
                    color = "🔴"

                st.markdown(
                    f"""
                **{sim['sim_id']}** {color} 负面率 {neg_rate:.1%}  
                {sim['test_content'][:50]}...
                """
                )
                st.caption(sim["created_at"])

            with col2:
                st.metric("触达", sim["total_reach"])

            with col3:
                if st.button("查看", key=f"view_{sim['sim_id']}"):
                    # 加载这个模拟结果
                    st.session_state.selected_simulation = sim
                    st.rerun()


def render_progress(world):
    """渲染模拟进度"""

    config = st.session_state.sim_config

    st.subheader("模拟进度")
    st.markdown(f"测试内容: {config['test_content'][:50]}...")

    st.divider()

    progress_text = st.empty()
    progress_bar = st.progress(0)
    step_log = st.container()

    async def run():
        llm = SentiSimLLM()
        steps_done = []

        def on_step(step, wave_size, new_posts):
            steps_done.append(
                f"Step {step}: 处理 {wave_size} 条反应, 产生 {new_posts} 条新帖子"
            )
            with step_log:
                for s in steps_done:
                    st.text(s)

            pct = (step + 1) / config["max_steps"]
            progress_bar.progress(min(pct, 1.0))
            progress_text.markdown(f"**Step {step + 1}/{config['max_steps']}**")

        result = await world.simulate(
            test_content=config["test_content"],
            max_steps=config["max_steps"],
            max_concurrent=config["max_concurrent"],
            reset_memory=config["reset_memory"],
            llm=llm,
            on_step_complete=on_step,
        )

        return result, llm.call_count

    try:
        result, call_count = asyncio.run(run())

        progress_bar.progress(1.0)
        progress_text.markdown("**✅ 模拟完成!**")

        st.success(f"模拟完成！触达 {result.total_reach} 人，LLM 调用 {call_count} 次")

        # 简要结果
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总触达", result.total_reach)
        with col2:
            st.metric("传播帖子", len(result.get_spreading_posts()))
        with col3:
            neg_count = len(result.get_negative_responses())
            neg_rate = neg_count / result.total_reach if result.total_reach > 0 else 0
            st.metric("负面率", f"{neg_rate:.1%}")
        with col4:
            st.metric("运行步数", result.steps_run)

        # 保存到 session 供分析页面使用
        st.session_state.current_simulation = result
        st.session_state.running_simulation = False

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 查看详细分析", use_container_width=True, type="primary"):
                st.rerun()
        with col2:
            if st.button("🔄 运行新模拟", use_container_width=True):
                st.session_state.running_simulation = False
                st.session_state.sim_config = None
                st.rerun()

    except Exception as e:
        st.error(f"模拟失败: {e}")
        st.session_state.running_simulation = False
        if st.button("重试"):
            st.rerun()
