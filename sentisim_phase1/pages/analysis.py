"""
页面: 分析结果
"""

import json
import streamlit.components.v1 as components
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from sentisim.visualization import NetworkVisualizer
from sentisim.visualization.network_viz import load_network_data


def load_simulation_from_path(
    sim_path: str,
) -> tuple[dict, dict, list[dict], list[dict]]:
    """从路径加载模拟数据"""
    sim_path = Path(sim_path)

    with open(sim_path / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(sim_path / "result.json", "r", encoding="utf-8") as f:
        result = json.load(f)

    responses = []
    with open(sim_path / "responses.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            responses.append(json.loads(line))

    posts = []
    posts_file = sim_path / "posts.json"
    if posts_file.exists():
        with open(posts_file, "r", encoding="utf-8") as f:
            posts = json.load(f)

    return config, result, responses, posts


def render():
    st.title("📊 分析结果")

    # 检查数据来源
    world = st.session_state.current_world
    selected_sim = st.session_state.get("selected_simulation")

    if not world:
        st.warning("请先在「数据管理」页面加载一个世界")
        return

    if not selected_sim:
        st.info("请在「数据管理」页面选择一个模拟记录查看")
        return

    # 从文件加载模拟数据
    config, result_stats, responses_data, posts_data = load_simulation_from_path(
        selected_sim["path"]
    )

    # 显示当前查看的模拟
    st.markdown(f"**模拟 ID:** `{selected_sim['sim_id']}`")
    st.markdown(f"**测试内容:** {config['test_content']}")
    st.divider()

    render_from_data(world, config, result_stats, responses_data, posts_data)


def render_from_data(
    world, config: dict, result_stats: dict, responses_data: list, posts_data: list
):
    """从数据渲染分析页面"""

    # 使用 tabs 组织内容
    tab_overview, tab_network, tab_details = st.tabs(
        ["📊 数据分析", "🕸️ 网络可视化", "💬 反应详情"]
    )

    with tab_overview:
        render_overview_tab(world, config, result_stats, responses_data, posts_data)

    with tab_network:
        render_network_tab(world, posts_data, responses_data)

    with tab_details:
        render_details_tab(world, responses_data)


def render_overview_tab(
    world, config: dict, result_stats: dict, responses_data: list, posts_data: list
):
    """渲染数据分析 tab"""

    # ===== 总览指标 =====
    st.subheader("总览指标")

    col1, col2, col3, col4 = st.columns(4)

    total_reach = result_stats["total_reach"]
    negative_count = result_stats["negative_count"]
    negative_rate = negative_count / total_reach if total_reach > 0 else 0

    with col1:
        st.metric("总触达", total_reach)
    with col2:
        st.metric("传播帖子", result_stats.get("spreading_posts", 0))
    with col3:
        st.metric("负面率", f"{negative_rate:.1%}")
    with col4:
        st.metric("运行步数", result_stats["steps_run"])

    st.divider()

    # ===== 时间演化 =====
    st.subheader("📈 时间演化")

    step_stats = aggregate_by_step(responses_data, result_stats["steps_run"])

    if step_stats:
        col1, col2 = st.columns(2)

        with col1:
            fig_neg = px.line(
                step_stats,
                x="step",
                y="negative_rate",
                title="负面率趋势",
                markers=True,
            )
            fig_neg.update_yaxes(
                tickformat=".0%",
                range=[0, max(0.5, max(s["negative_rate"] for s in step_stats) * 1.2)],
            )
            fig_neg.update_layout(xaxis_title="Step", yaxis_title="负面率")
            st.plotly_chart(fig_neg, use_container_width=True)

        with col2:
            fig_reach = px.line(
                step_stats,
                x="step",
                y="cumulative_reach",
                title="累计触达",
                markers=True,
            )
            fig_reach.update_layout(xaxis_title="Step", yaxis_title="累计人数")
            st.plotly_chart(fig_reach, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig_wave = px.bar(
                step_stats,
                x="step",
                y="reach",
                title="每步触达人数",
            )
            fig_wave.update_layout(xaxis_title="Step", yaxis_title="人数")
            st.plotly_chart(fig_wave, use_container_width=True)

        with col4:
            fig_spread = px.bar(
                step_stats,
                x="step",
                y="spread_count",
                title="每步新增传播",
            )
            fig_spread.update_layout(xaxis_title="Step", yaxis_title="新帖子数")
            st.plotly_chart(fig_spread, use_container_width=True)

    st.divider()

    # ===== 分布分析 =====
    st.subheader("📊 分布分析")

    col1, col2 = st.columns(2)

    with col1:
        action_counts = result_stats["action_counts"]
        action_df = pd.DataFrame(
            [{"行动": k, "数量": v} for k, v in action_counts.items()]
        )
        fig_action = px.pie(
            action_df,
            values="数量",
            names="行动",
            title="行动分布",
        )
        st.plotly_chart(fig_action, use_container_width=True)

    with col2:
        emotion_counts = result_stats["emotion_counts"]
        emotion_sorted = sorted(emotion_counts.items(), key=lambda x: -x[1])[:10]
        emotion_df = pd.DataFrame([{"情绪": k, "数量": v} for k, v in emotion_sorted])
        fig_emotion = px.bar(
            emotion_df,
            x="数量",
            y="情绪",
            orientation="h",
            title="情绪分布 (Top 10)",
        )
        fig_emotion.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_emotion, use_container_width=True)

    st.divider()

    # ===== 人群对比 =====
    st.subheader("👥 人群对比")

    persona_stats = aggregate_by_persona(responses_data)

    if persona_stats:
        persona_df = pd.DataFrame(persona_stats)

        fig_persona = px.bar(
            persona_df.sort_values("negative_rate", ascending=True),
            x="negative_rate",
            y="persona",
            orientation="h",
            title="各人群负面率对比",
            text=persona_df.sort_values("negative_rate", ascending=True)[
                "negative_rate"
            ].apply(lambda x: f"{x:.1%}"),
        )
        fig_persona.update_xaxes(tickformat=".0%")
        fig_persona.update_layout(yaxis_title="", xaxis_title="负面率")
        st.plotly_chart(fig_persona, use_container_width=True)

        with st.expander("查看人群详细数据"):
            display_df = persona_df.copy()
            display_df["negative_rate"] = display_df["negative_rate"].apply(
                lambda x: f"{x:.1%}"
            )
            display_df["spread_rate"] = display_df["spread_rate"].apply(
                lambda x: f"{x:.1%}"
            )
            display_df.columns = [
                "人群",
                "总数",
                "负面数",
                "负面率",
                "传播数",
                "传播率",
            ]
            st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_network_tab(world, posts_data: list, responses_data: list):
    """渲染网络可视化 tab"""

    st.subheader("🕸️ 网络与传播可视化")

    # 加载网络数据
    world_path = Path("data/worlds") / world.meta.world_id
    try:
        network_data, users_data = load_network_data(str(world_path))
    except FileNotFoundError as e:
        st.error(f"无法加载网络数据: {e}")
        return

    viz = NetworkVisualizer(network_data, users_data)

    # 可视化选项
    viz_type = st.radio(
        "选择可视化类型",
        ["传播过程", "传播树", "网络拓扑"],
        horizontal=True,
    )

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("**图例**")
        if viz_type == "传播过程":
            st.markdown("""
            - ⭐ **紫色星** - 品牌账号
            - 🟢 **绿色** - 传播用户
            - 🔴 **红色** - 负面反应
            - 🟡 **黄色** - 点赞
            - ⚪ **灰色** - 忽略/其他
            """)

            # 步数选择
            max_step = max(r["step"] for r in responses_data) if responses_data else 0
            show_step = st.slider(
                "显示到第几步",
                min_value=0,
                max_value=max_step,
                value=max_step,
                help="拖动查看传播过程的演化",
            )
        elif viz_type == "传播树":
            st.markdown("""
            - ⭐ **紫色星** - 品牌发布
            - 🟢 **绿色** - 转发
            - 🔵 **蓝色** - 转发评论
            - 🔴 **红色** - 负面反应
            """)
        else:  # 网络拓扑
            st.markdown("""
            - ⭐ **紫色星** - 品牌账号
            - 🔴 **红色** - KOL
            - 🟠 **橙色** - 活跃用户
            - 🔵 **蓝色** - 普通用户
            - ⚪ **灰色** - 沉默用户
            """)

        show_labels = st.checkbox("显示标签", value=False)

    with col1:
        with st.spinner("生成可视化..."):
            if viz_type == "传播过程":
                html = viz.render_propagation(
                    posts_data,
                    responses_data,
                    step=show_step,
                    height="550px",
                    show_labels=show_labels,
                )
            elif viz_type == "传播树":
                html = viz.render_propagation_tree(
                    posts_data,
                    responses_data,
                    height="550px",
                )
            else:  # 网络拓扑
                html = viz.render_network_topology(
                    height="550px",
                    show_labels=show_labels,
                    physics=True,
                )

            components.html(html, height=600, scrolling=True)

    # 传播统计
    if viz_type == "传播过程":
        st.divider()
        st.subheader("📊 传播统计")

        # 统计传播链
        spread_posts = [p for p in posts_data if p["post_type"] in ["forward", "forward_comment"]]
        forward_count = sum(1 for p in spread_posts if p["post_type"] == "forward")
        comment_count = sum(1 for p in spread_posts if p["post_type"] == "forward_comment")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总传播帖子", len(spread_posts))
        with col2:
            st.metric("纯转发", forward_count)
        with col3:
            st.metric("转发评论", comment_count)


def render_details_tab(world, responses_data: list):
    """渲染反应详情 tab"""

    st.subheader("💬 反应详情")

    # 筛选器
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        steps = sorted(set(r["step"] for r in responses_data))
        selected_step = st.selectbox("Step", ["全部"] + steps, key="detail_step")

    with col2:
        personas = sorted(set(r["user_persona"] for r in responses_data))
        selected_persona = st.selectbox("人群", ["全部"] + personas, key="detail_persona")

    with col3:
        selected_sentiment = st.selectbox("情感", ["全部", "负面", "非负面"], key="detail_sentiment")

    with col4:
        actions = sorted(set(r["response"]["action"] for r in responses_data))
        selected_action = st.selectbox("行动", ["全部"] + actions, key="detail_action")

    # 筛选
    filtered = responses_data
    if selected_step != "全部":
        filtered = [r for r in filtered if r["step"] == selected_step]
    if selected_persona != "全部":
        filtered = [r for r in filtered if r["user_persona"] == selected_persona]
    if selected_sentiment == "负面":
        filtered = [r for r in filtered if r["response"]["is_negative"]]
    elif selected_sentiment == "非负面":
        filtered = [r for r in filtered if not r["response"]["is_negative"]]
    if selected_action != "全部":
        filtered = [r for r in filtered if r["response"]["action"] == selected_action]

    st.markdown(f"共 **{len(filtered)}** 条反应")

    # 显示反应卡片
    for i, record in enumerate(filtered[:50]):
        resp = record["response"]

        emoji = (
            "😠"
            if resp["is_negative"]
            else "😊" if resp["emotion"] in ["开心", "期待", "喜欢"] else "😐"
        )
        spread_icon = "📢" if resp["will_spread"] else ""

        with st.expander(
            f"{spread_icon} [{record['step']}] {record['user_persona']} | {emoji} {resp['emotion']} | {resp['action']}"
        ):
            st.markdown(f"**用户:** {record['user_id']}")
            st.markdown(f"**第一反应:** {resp['first_reaction']}")
            st.markdown(f"**情绪:** {resp['emotion']}")
            st.markdown(f"**印象变化:** {resp['impression_change']}")
            st.markdown(f"**行动:** {resp['action']}")
            if resp["content"]:
                st.markdown(f"**发布内容:** {resp['content']}")

            user = world.get_user(record["user_id"])
            if user:
                st.divider()
                st.markdown(f"**用户画像:** {user.profile}")
                if user.memory:
                    st.markdown(f"**记忆:** {', '.join(user.memory)}")

    if len(filtered) > 50:
        st.info(f"仅显示前 50 条，共 {len(filtered)} 条")


def aggregate_by_step(responses_data: list, steps_run: int) -> list[dict]:
    """按 step 聚合统计"""
    by_step = {}
    for r in responses_data:
        step = r["step"]
        if step not in by_step:
            by_step[step] = []
        by_step[step].append(r)

    stats = []
    cumulative = 0

    for step in range(steps_run):
        records = by_step.get(step, [])
        reach = len(records)
        cumulative += reach

        negative_count = sum(1 for r in records if r["response"]["is_negative"])
        spread_count = sum(1 for r in records if r["response"]["will_spread"])

        stats.append(
            {
                "step": step,
                "reach": reach,
                "cumulative_reach": cumulative,
                "negative_count": negative_count,
                "negative_rate": negative_count / reach if reach > 0 else 0,
                "spread_count": spread_count,
            }
        )

    return stats


def aggregate_by_persona(responses_data: list) -> list[dict]:
    """按人群聚合统计"""
    by_persona = {}
    for r in responses_data:
        persona = r["user_persona"]
        if persona not in by_persona:
            by_persona[persona] = []
        by_persona[persona].append(r)

    stats = []
    for persona, records in by_persona.items():
        count = len(records)
        negative_count = sum(1 for r in records if r["response"]["is_negative"])
        spread_count = sum(1 for r in records if r["response"]["will_spread"])

        stats.append(
            {
                "persona": persona,
                "count": count,
                "negative_count": negative_count,
                "negative_rate": negative_count / count if count > 0 else 0,
                "spread_count": spread_count,
                "spread_rate": spread_count / count if count > 0 else 0,
            }
        )

    return stats
