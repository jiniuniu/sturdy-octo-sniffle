"""
页面1: 数据管理 - 加载世界和选择模拟
"""

import json
import shutil
from pathlib import Path

import streamlit as st
from sentisim.world import World

WORLDS_DIR = Path("data/worlds")


def get_worlds_list() -> list[dict]:
    """获取所有已保存的世界"""
    if not WORLDS_DIR.exists():
        return []

    worlds = []
    for world_dir in sorted(WORLDS_DIR.iterdir(), reverse=True):
        if world_dir.is_dir() and (world_dir / "meta.json").exists():
            with open(world_dir / "meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)

            sim_dir = world_dir / "simulations"
            sim_count = len(list(sim_dir.glob("sim_*"))) if sim_dir.exists() else 0

            worlds.append(
                {
                    "path": str(world_dir),
                    "world_id": meta["world_id"],
                    "user_count": meta["user_count"],
                    "persona_count": meta["persona_count"],
                    "created_at": meta["created_at"],
                    "brand_name": meta["brand_context"]["brand_name"],
                    "sim_count": sim_count,
                }
            )

    return worlds


def get_simulation_list(world_id: str) -> list[dict]:
    """获取世界的所有模拟"""
    sim_dir = WORLDS_DIR / world_id / "simulations"

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
                    "steps_run": result["steps_run"],
                }
            )

    return simulations


def load_world(world_path: str):
    """加载世界到 session state"""
    try:
        world = World.load(world_path)
        st.session_state.current_world = world
        st.session_state.selected_simulation = None
        st.success(f"已加载: {world.meta.world_id}")
        st.rerun()
    except Exception as e:
        st.error(f"加载失败: {e}")


def delete_world(world_path: str):
    """删除世界"""
    try:
        shutil.rmtree(world_path)
        if st.session_state.current_world:
            current_path = str(
                WORLDS_DIR / st.session_state.current_world.meta.world_id
            )
            if current_path == world_path:
                st.session_state.current_world = None
                st.session_state.selected_simulation = None
        st.success("已删除")
        st.rerun()
    except Exception as e:
        st.error(f"删除失败: {e}")


def render():
    st.title("📂 数据管理")

    # 两栏布局
    col_worlds, col_sims = st.columns([1, 1])

    # ===== 左侧：世界列表 =====
    with col_worlds:
        st.subheader("虚拟世界")

        worlds = get_worlds_list()

        if not worlds:
            st.info("还没有创建任何世界\n\n使用 `python main.py` 创建")
            return

        for world in worlds:
            is_current = (
                st.session_state.current_world
                and st.session_state.current_world.meta.world_id == world["world_id"]
            )

            with st.container():
                # 标题
                if is_current:
                    st.markdown(f"#### ✅ {world['world_id']}")
                else:
                    st.markdown(f"#### {world['world_id']}")

                # 信息
                st.caption(
                    f"🏷️ {world['brand_name']} | "
                    f"👥 {world['user_count']}人 | "
                    f"🎭 {world['persona_count']}种人群 | "
                    f"📊 {world['sim_count']}次模拟"
                )

                # 按钮
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if not is_current:
                        if st.button("加载", key=f"load_{world['world_id']}", use_container_width=True):
                            load_world(world["path"])
                    else:
                        st.button("已加载", key=f"loaded_{world['world_id']}", disabled=True, use_container_width=True)

                with btn_col2:
                    if st.button("删除", key=f"del_{world['world_id']}", use_container_width=True):
                        st.session_state[f"confirm_del_{world['world_id']}"] = True

                # 删除确认
                if st.session_state.get(f"confirm_del_{world['world_id']}"):
                    st.warning("确定删除？")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("确认", key=f"yes_{world['world_id']}", type="primary"):
                            delete_world(world["path"])
                    with c2:
                        if st.button("取消", key=f"no_{world['world_id']}"):
                            st.session_state[f"confirm_del_{world['world_id']}"] = False
                            st.rerun()

                st.divider()

    # ===== 右侧：模拟列表 =====
    with col_sims:
        st.subheader("模拟记录")

        if not st.session_state.current_world:
            st.info("请先加载一个世界")
            return

        world = st.session_state.current_world
        simulations = get_simulation_list(world.meta.world_id)

        if not simulations:
            st.info("该世界还没有运行过模拟\n\n使用 `python main.py` 运行")
            return

        for sim in simulations:
            is_selected = (
                st.session_state.selected_simulation
                and st.session_state.selected_simulation["sim_id"] == sim["sim_id"]
            )

            with st.container():
                # 负面率颜色标识
                neg_rate = sim["negative_rate"]
                if neg_rate < 0.15:
                    indicator = "🟢"
                elif neg_rate < 0.30:
                    indicator = "🟡"
                else:
                    indicator = "🔴"

                # 标题
                if is_selected:
                    st.markdown(f"#### ✅ {sim['sim_id']}")
                else:
                    st.markdown(f"#### {sim['sim_id']}")

                # 指标
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("触达", sim["total_reach"])
                with m2:
                    st.metric("负面率", f"{indicator} {neg_rate:.1%}")
                with m3:
                    st.metric("步数", sim["steps_run"])

                # 内容预览
                st.caption(f"📝 {sim['test_content'][:60]}...")

                # 选择按钮
                if not is_selected:
                    if st.button("查看分析", key=f"select_{sim['sim_id']}", use_container_width=True, type="primary"):
                        st.session_state.selected_simulation = sim
                        st.rerun()
                else:
                    st.button("已选择", key=f"selected_{sim['sim_id']}", disabled=True, use_container_width=True)

                st.divider()
