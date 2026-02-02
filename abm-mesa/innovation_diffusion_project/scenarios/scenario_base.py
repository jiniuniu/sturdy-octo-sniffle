"""
场景配置基类
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ScenarioConfig:
    """
    场景配置类
    
    Attributes:
        name: 场景名称
        description: 场景描述
        params: 模型参数字典
        hypothesis: 研究假设（可选）
        expected_outcome: 预期结果（可选）
    """
    name: str
    description: str
    params: Dict[str, Any]
    hypothesis: Optional[str] = None
    expected_outcome: Optional[str] = None
    
    def get_model_params(self) -> Dict[str, Any]:
        """返回模型参数"""
        return self.params
    
    def get_model_params_config(self) -> Dict[str, Any]:
        """
        返回 SolaraViz 的模型参数配置
        用于生成 UI 控件
        """
        return {
            "n_consumers": {
                "type": "SliderInt",
                "value": self.params.get("n_consumers", 100),
                "label": "消费者数量",
                "min": 50,
                "max": 200,
                "step": 10,
            },
            "k_neighbors": {
                "type": "SliderInt",
                "value": self.params.get("k_neighbors", 4),
                "label": "邻居数量",
                "min": 2,
                "max": 10,
                "step": 1,
            },
            "p_rewire": {
                "type": "SliderFloat",
                "value": self.params.get("p_rewire", 0.3),
                "label": "重连概率",
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
            },
            "initial_buyers_pct": {
                "type": "SliderFloat",
                "value": self.params.get("initial_buyers_pct", 0.05),
                "label": "初始购买者比例",
                "min": 0.0,
                "max": 0.15,
                "step": 0.01,
            },
        }
