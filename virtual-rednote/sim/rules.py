"""
规则引擎：向量匹配 + 行为概率采样
"""

import numpy as np

from .models import ActionType, Agent, AgentTier, Content


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def compute_action(
    agent: Agent, content: Content, social_pressure: float
) -> ActionType:
    """
    基于向量匹配度计算行为概率并采样。

    social_pressure：agent 关注列表中已转发此内容的比例。
    KOL 的转发在 social_pressure 计算中权重更高（见 engine）。

    p_like    = match * activity * 0.6
    p_comment = match * expressiveness * 0.3
    p_repost  = match * sharing * (1 + social_pressure) * 0.2
    """
    match = cosine_similarity(agent.vector, content.vector)

    p_like = match * agent.activity * 0.6
    p_comment = match * agent.expressiveness * 0.3
    p_repost = match * agent.sharing * (1.0 + social_pressure) * 0.2

    r = np.random.random()
    if r < p_repost:
        return ActionType.REPOST
    elif r < p_repost + p_comment:
        return ActionType.COMMENT
    elif r < p_repost + p_comment + p_like:
        return ActionType.LIKE
    else:
        return ActionType.IGNORE
