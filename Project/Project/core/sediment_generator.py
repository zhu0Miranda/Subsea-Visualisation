import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional, Any
import random

class SedimentModelGenerator:
    """沉积层地质模型生成器"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            # 修复编码问题
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
            except UnicodeDecodeError:
                try:
                    with open(config_path, 'r', encoding='gbk') as f:
                        self.config = yaml.safe_load(f)
                except Exception as e:
                    print(f"无法读取配置文件: {e}，使用默认配置")
                    self._set_default_config()
        else:
            # 无配置文件时使用默认配置
            self._set_default_config()
        
        self.sediment_types = self.config['sediment_types']
    
    def _set_default_config(self):
        """设置默认配置"""
        self.config = {
            'simulation': {
                'sample_rate': 100000,
                'trace_length': 1024
            },
            'sediment': {
                'min_layers': 1,
                'max_layers': 8,
                'thickness_range': [0.1, 5.0]
            },
            'sediment_types': {
                'clay': {'density': [1500, 1700], 'velocity': [1470, 1550]},
                'silt': {'density': [1650, 1850], 'velocity': [1550, 1650]},
                'sand': {'density': [1800, 2100], 'velocity': [1650, 1850]},
                'gravel': {'density': [1900, 2200], 'velocity': [1800, 2200]}
            }
        }
    
    # 其他方法保持不变...
    # [之前的 generate_layered_model, add_geological_features 等方法]