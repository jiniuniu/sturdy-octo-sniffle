"""
页面1: 世界管理
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


def load_world(world_path: str):
    """加载世界到 session state"""
    try:
        world = World.load(world_path)
        st.session_state.current_world = world
        st.session_state.current_simulation = None
        st.success(f"已加载: {world.meta.world_id}")
        st.rerun()
    except Exception as e:
        st.error(f"加载失败: {e}")


def delete_world(world_path: str):
    """删除世界"""
    try:
        shutil.rmtree(world_path)
        # 如果删除的是当前加载的世界，清除 session
        if st.session_state.current_world:
            current_path = str(
                WORLDS_DIR / st.session_state.current_world.meta.world_id
            )
            if current_path == world_path:
                st.session_state.current_world = None
                st.session_state.current_simulation = None
        st.success("已删除")
        st.rerun()
    except Exception as e:
        st.error(f"删除失败: {e}")


def render():
    st.title("🏠 世界管理")
    st.markdown("管理已创建的虚拟世界")

    worlds = get_worlds_list()

    if not worlds:
        st.info("还没有创建任何世界，点击左侧「创建世界」开始")
        return

    st.markdown(f"共有 **{len(worlds)}** 个世界")
    st.divider()

    for world in worlds:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                # 标题和信息
                is_current = (
                    st.session_state.current_world
                    and st.session_state.current_world.meta.world_id
                    == world["world_id"]
                )

                if is_current:
                    st.markdown(f"### 📂 {world['world_id']} ✅")
                else:
                    st.markdown(f"### 📁 {world['world_id']}")

                st.markdown(
                    f"**品牌:** {world['brand_name']} | "
                    f"**用户:** {world['user_count']} | "
                    f"**人群:** {world['persona_count']}种 | "
                    f"**模拟:** {world['sim_count']}次"
                )
                st.caption(f"创建时间: {world['created_at']}")

            with col2:
                if st.button(
                    "加载", key=f"load_{world['world_id']}", use_container_width=True
                ):
                    load_world(world["path"])

            with col3:
                if st.button(
                    "删除",
                    key=f"del_{world['world_id']}",
                    type="secondary",
                    use_container_width=True,
                ):
                    st.session_state[f"confirm_del_{world['world_id']}"] = True

            # 删除确认
            if st.session_state.get(f"confirm_del_{world['world_id']}"):
                st.warning("确定要删除这个世界吗？此操作不可恢复。")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button(
                        "确认删除",
                        key=f"confirm_yes_{world['world_id']}",
                        type="primary",
                    ):
                        delete_world(world["path"])
                with col_no:
                    if st.button("取消", key=f"confirm_no_{world['world_id']}"):
                        st.session_state[f"confirm_del_{world['world_id']}"] = False
                        st.rerun()

            st.divider()
