"""
SentiSim Streamlit 应用 - 数据可视化

运行方式:
    cd sentisim_phase1
    streamlit run app.py

生成数据请使用 terminal:
    python main.py
"""

import streamlit as st

st.set_page_config(
    page_title="SentiSim 舆情模拟",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 初始化 session state
if "current_world" not in st.session_state:
    st.session_state.current_world = None
if "selected_simulation" not in st.session_state:
    st.session_state.selected_simulation = None


def main():
    st.sidebar.title("🎭 SentiSim")
    st.sidebar.markdown("舆情风险推演系统")
    st.sidebar.divider()

    # 导航
    page = st.sidebar.radio(
        "导航",
        ["📂 数据管理", "📊 分析结果"],
        label_visibility="collapsed",
    )

    # 显示当前加载的世界
    if st.session_state.current_world:
        st.sidebar.divider()
        st.sidebar.success(f"当前世界:\n{st.session_state.current_world.meta.world_id}")

        if st.session_state.selected_simulation:
            st.sidebar.info(
                f"当前模拟:\n{st.session_state.selected_simulation['sim_id']}"
            )

    # 路由
    if page == "📂 数据管理":
        from pages import world_manager

        world_manager.render()
    elif page == "📊 分析结果":
        from pages import analysis

        analysis.render()


if __name__ == "__main__":
    main()
