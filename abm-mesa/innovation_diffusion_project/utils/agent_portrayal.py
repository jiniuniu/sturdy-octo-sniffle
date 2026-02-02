"""
Agent 可视化配置
"""
from models import DIFFUSION_CONFIG, BuyingState


def agent_portrayal(agent):
    """
    定义 agent 的视觉表现 - 根据类型和状态
    
    Args:
        agent: ConsumerAgent 实例
        
    Returns:
        dict: 包含 color, size, alpha 等可视化参数
    """
    config = DIFFUSION_CONFIG[agent.consumer_type]
    
    if agent.state == BuyingState.BOUGHT:
        # 已购买：使用类型颜色，较大尺寸
        return {
            "color": config["color"],
            "size": 20,
        }
    else:
        # 未购买：使用类型颜色但透明，较小尺寸
        return {
            "color": config["color"],
            "size": 8,
            "alpha": 0.3,  # 半透明
        }
