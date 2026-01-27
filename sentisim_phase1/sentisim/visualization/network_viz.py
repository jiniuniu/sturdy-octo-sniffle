"""
网络可视化模块 - 使用 Pyvis 渲染网络图和传播过程
"""

import json
import tempfile
from pathlib import Path
from typing import Optional

import networkx as nx
from pyvis.network import Network


class NetworkVisualizer:
    """网络可视化器"""

    # 影响力层级对应的颜色和大小
    INFLUENCE_COLORS = {
        "kol": "#e74c3c",  # 红色 - KOL
        "active": "#f39c12",  # 橙色 - 活跃用户
        "normal": "#3498db",  # 蓝色 - 普通用户
        "silent": "#95a5a6",  # 灰色 - 沉默用户
    }

    INFLUENCE_SIZES = {
        "kol": 30,
        "active": 20,
        "normal": 12,
        "silent": 8,
    }

    # 情绪对应的颜色
    EMOTION_COLORS = {
        "positive": "#2ecc71",  # 绿色 - 正面
        "neutral": "#3498db",  # 蓝色 - 中性
        "negative": "#e74c3c",  # 红色 - 负面
    }

    def __init__(self, network_data: dict, users_data: Optional[list] = None):
        """
        初始化可视化器

        Args:
            network_data: 网络数据，包含 edges, user_ids, brand_account_id
            users_data: 用户数据列表（可选），用于获取影响力等信息
        """
        self.network_data = network_data
        self.users_dict = {}
        if users_data:
            self.users_dict = {u["user_id"]: u for u in users_data}

        # 构建 NetworkX 图
        self.graph = nx.DiGraph()
        for edge in network_data["edges"]:
            self.graph.add_edge(edge[0], edge[1])

        # 添加所有节点（包括可能没有边的节点）
        for uid in network_data.get("user_ids", []):
            if uid not in self.graph:
                self.graph.add_node(uid)

        # 添加品牌账号
        brand_id = network_data.get("brand_account_id", "brand_official")
        if brand_id not in self.graph:
            self.graph.add_node(brand_id)

        self.brand_account_id = brand_id

    def render_network_topology(
        self,
        height: str = "600px",
        width: str = "100%",
        show_labels: bool = False,
        physics: bool = True,
    ) -> str:
        """
        渲染网络拓扑图

        Args:
            height: 图的高度
            width: 图的宽度
            show_labels: 是否显示节点标签
            physics: 是否启用物理引擎（布局动画）

        Returns:
            HTML 字符串
        """
        net = Network(
            height=height,
            width=width,
            bgcolor="#ffffff",
            font_color="#333333",
            directed=True,
            notebook=False,
        )

        # 配置物理引擎
        if physics:
            net.barnes_hut(
                gravity=-8000,
                central_gravity=0.3,
                spring_length=100,
                spring_strength=0.01,
            )
        else:
            net.toggle_physics(False)

        # 添加节点
        for node in self.graph.nodes():
            if node == self.brand_account_id:
                # 品牌账号特殊样式
                net.add_node(
                    node,
                    label="品牌" if show_labels else "",
                    color="#9b59b6",
                    size=40,
                    shape="star",
                    title=f"品牌官方账号\n粉丝数: {self.graph.in_degree(node)}",
                )
            else:
                # 普通用户
                user_info = self.users_dict.get(node, {})
                influence = user_info.get("influence_level", "normal")
                persona = user_info.get("persona_type", "未知")

                color = self.INFLUENCE_COLORS.get(influence, "#3498db")
                size = self.INFLUENCE_SIZES.get(influence, 12)

                title = f"用户: {node}\n类型: {persona}\n影响力: {influence}\n粉丝: {self.graph.in_degree(node)}"

                net.add_node(
                    node,
                    label=node[-4:] if show_labels else "",
                    color=color,
                    size=size,
                    title=title,
                )

        # 添加边
        for edge in self.graph.edges():
            net.add_edge(edge[0], edge[1], color="#cccccc", width=0.5)

        # 生成 HTML
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            net.save_graph(f.name)
            with open(f.name, "r", encoding="utf-8") as rf:
                html = rf.read()

        return html

    def render_propagation(
        self,
        posts_data: list,
        responses_data: list,
        step: Optional[int] = None,
        height: str = "600px",
        width: str = "100%",
        show_labels: bool = False,
        show_follow_edges: bool = True,
    ) -> str:
        """
        渲染传播过程图

        Args:
            posts_data: 帖子数据列表
            responses_data: 响应数据列表
            step: 指定步数（None 表示全部）
            height: 图的高度
            width: 图的宽度
            show_labels: 是否显示节点标签
            show_follow_edges: 是否显示关注关系边

        Returns:
            HTML 字符串
        """
        net = Network(
            height=height,
            width=width,
            bgcolor="#ffffff",
            font_color="#333333",
            directed=True,
            notebook=False,
        )

        net.barnes_hut(
            gravity=-5000,
            central_gravity=0.5,
            spring_length=150,
            spring_strength=0.02,
        )

        # 筛选响应数据
        if step is not None:
            responses = [r for r in responses_data if r["step"] <= step]
        else:
            responses = responses_data

        # 构建用户响应映射
        user_responses = {}
        for r in responses:
            uid = r["user_id"]
            if uid not in user_responses or r["step"] > user_responses[uid]["step"]:
                user_responses[uid] = r

        # 构建帖子映射
        posts_dict = {p["post_id"]: p for p in posts_data}

        # 找出参与传播的用户和传播边
        spreaders = set()  # 产生传播行为的用户
        propagation_edges = []

        for post in posts_data:
            if post["post_type"] in ["forward", "forward_comment"]:
                author = post["author_id"]
                original_post_id = post.get("original_post_id")
                if original_post_id and original_post_id in posts_dict:
                    original_author = posts_dict[original_post_id]["author_id"]
                    spreaders.add(author)
                    spreaders.add(original_author)
                    propagation_edges.append(
                        (original_author, author, post["post_type"])
                    )

        # 添加品牌账号（起点）
        spreaders.add(self.brand_account_id)

        # 只显示参与传播的节点（或所有被触达的用户）
        reached_users = set(r["user_id"] for r in responses)
        display_users = spreaders | reached_users

        # 添加节点
        for node in display_users:
            if node == self.brand_account_id:
                net.add_node(
                    node,
                    label="品牌" if show_labels else "",
                    color="#9b59b6",
                    size=45,
                    shape="star",
                    title="品牌官方账号",
                )
            else:
                user_resp = user_responses.get(node)
                user_info = self.users_dict.get(node, {})
                influence = user_info.get("influence_level", "normal")
                persona = user_info.get("persona_type", "未知")

                # 根据响应确定颜色
                if user_resp:
                    is_negative = user_resp["response"].get("is_negative", False)
                    will_spread = user_resp["response"].get("will_spread", False)
                    action = user_resp["response"].get("action", "ignore")
                    emotion = user_resp["response"].get("emotion", "")

                    if is_negative:
                        color = "#e74c3c"  # 红色 - 负面
                    elif will_spread:
                        color = "#2ecc71"  # 绿色 - 传播
                    elif action == "like":
                        color = "#f1c40f"  # 黄色 - 点赞
                    else:
                        color = "#95a5a6"  # 灰色 - 忽略

                    title = f"用户: {node}\n类型: {persona}\n情绪: {emotion}\n行动: {action}"
                else:
                    color = "#bdc3c7"
                    title = f"用户: {node}\n类型: {persona}"

                size = self.INFLUENCE_SIZES.get(influence, 12) + 5

                net.add_node(
                    node,
                    label=node[-4:] if show_labels else "",
                    color=color,
                    size=size,
                    title=title,
                )

        # 记录已添加的边，避免重复
        added_edges = set()

        # 添加传播边（实线、粗、彩色）
        for src, dst, ptype in propagation_edges:
            if src in display_users and dst in display_users:
                edge_color = "#2ecc71" if ptype == "forward" else "#3498db"
                net.add_edge(
                    src,
                    dst,
                    color=edge_color,
                    width=3,
                    title=f"{'转发' if ptype == 'forward' else '转发评论'}",
                    arrows="to",
                )
                added_edges.add((src, dst))

        # 添加关注关系边（虚线、细、灰色）
        if show_follow_edges:
            # 品牌粉丝关系
            brand_followers = set(self.network_data.get("brand_followers", []))
            for uid in brand_followers:
                if (
                    uid in display_users
                    and (self.brand_account_id, uid) not in added_edges
                ):
                    net.add_edge(
                        self.brand_account_id,
                        uid,
                        color="#d5d5d5",
                        width=0.8,
                        dashes=True,
                        title="关注品牌",
                        arrows="to",
                    )
                    added_edges.add((self.brand_account_id, uid))

            # 传播用户的粉丝关系：spreader -> 他的粉丝（在display_users中的）
            for spreader in spreaders:
                if spreader == self.brand_account_id:
                    continue
                # 获取该用户的粉丝（谁关注了他）
                followers = self.graph.predecessors(spreader)
                for follower in followers:
                    if (
                        follower in display_users
                        and (spreader, follower) not in added_edges
                    ):
                        net.add_edge(
                            spreader,
                            follower,
                            color="#d5d5d5",
                            width=0.8,
                            dashes=True,
                            title=f"关注 {spreader[-4:]}",
                            arrows="to",
                        )
                        added_edges.add((spreader, follower))

        # 生成 HTML
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            net.save_graph(f.name)
            with open(f.name, "r", encoding="utf-8") as rf:
                html = rf.read()

        return html

    def render_propagation_tree(
        self,
        posts_data: list,
        responses_data: list,
        height: str = "500px",
        width: str = "100%",
    ) -> str:
        """
        渲染传播树（层级结构）

        Args:
            posts_data: 帖子数据列表
            responses_data: 响应数据列表
            height: 图的高度
            width: 图的宽度

        Returns:
            HTML 字符串
        """
        net = Network(
            height=height,
            width=width,
            bgcolor="#ffffff",
            font_color="#333333",
            directed=True,
            notebook=False,
            layout="hierarchical",
        )

        # 配置层级布局
        net.set_options(
            """
        {
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "UD",
                    "sortMethod": "directed",
                    "levelSeparation": 100,
                    "nodeSpacing": 150
                }
            },
            "physics": {
                "enabled": false
            }
        }
        """
        )

        # 构建帖子映射和传播关系
        posts_dict = {p["post_id"]: p for p in posts_data}
        user_responses = {r["user_id"]: r for r in responses_data}

        # 找到根帖子（品牌原创）
        root_posts = [p for p in posts_data if p["post_type"] == "brand_original"]

        if not root_posts:
            return "<p>没有传播数据</p>"

        # 添加品牌节点
        net.add_node(
            self.brand_account_id,
            label="品牌发布",
            color="#9b59b6",
            size=35,
            shape="star",
            level=0,
        )

        # 构建树结构
        added_nodes = {self.brand_account_id}

        for post in posts_data:
            post_id = post["post_id"]
            author = post["author_id"]
            post_type = post["post_type"]

            if post_type == "brand_original":
                continue

            # 获取用户响应信息
            user_resp = user_responses.get(author, {})
            resp_data = user_resp.get("response", {})
            emotion = resp_data.get("emotion", "")
            is_negative = resp_data.get("is_negative", False)
            step = user_resp.get("step", 0)

            # 确定节点颜色
            if is_negative:
                color = "#e74c3c"
            else:
                color = "#2ecc71" if post_type == "forward" else "#3498db"

            # 节点标签
            label = f"{author[-4:]}\n{emotion}" if emotion else author[-4:]

            # 添加节点
            if author not in added_nodes:
                net.add_node(
                    author,
                    label=label,
                    color=color,
                    size=20,
                    level=step + 1,
                    title=f"用户: {author}\n类型: {post_type}\n情绪: {emotion}",
                )
                added_nodes.add(author)

            # 添加边
            original_post_id = post.get("original_post_id")
            if original_post_id and original_post_id in posts_dict:
                original_author = posts_dict[original_post_id]["author_id"]
                if original_author in added_nodes:
                    edge_label = "转发" if post_type == "forward" else "转评"
                    net.add_edge(
                        original_author,
                        author,
                        title=edge_label,
                        color=color,
                        width=2,
                    )

        # 生成 HTML
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            net.save_graph(f.name)
            with open(f.name, "r", encoding="utf-8") as rf:
                html = rf.read()

        return html


def load_network_data(world_path: str) -> tuple[dict, list]:
    """
    从世界路径加载网络数据和用户数据

    Args:
        world_path: 世界目录路径

    Returns:
        (network_data, users_data)
    """
    world_path = Path(world_path)

    with open(world_path / "network.json", "r", encoding="utf-8") as f:
        network_data = json.load(f)

    with open(world_path / "users.json", "r", encoding="utf-8") as f:
        users_data = json.load(f)

    return network_data, users_data
