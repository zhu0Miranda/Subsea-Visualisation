#-*- coding: GB18030 -*-
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional, Any
import random

class SedimentModelGenerator:
    """沉积层地质模型生成器"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            # 使用GB18030编码读取配置文件
            try:
                with open(config_path, 'r', encoding='gb18030') as f:
                    self.config = yaml.safe_load(f)
                print("SedimentGenerator: 成功使用GB18030编码读取配置文件")
            except Exception as e:
                print(f"SedimentGenerator: 使用GB18030编码读取失败: {e}")
                # 尝试其他编码
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        self.config = yaml.safe_load(f)
                    print("SedimentGenerator: 成功使用UTF-8编码读取配置文件")
                except Exception as e2:
                    print(f"SedimentGenerator: 使用UTF-8编码读取也失败: {e2}")
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
    
    def generate_layered_model(self) -> Dict:
        """生成随机层状沉积模型"""
        n_layers = random.randint(
            self.config['sediment']['min_layers'],
            self.config['sediment']['max_layers']
        )
        
        layers = []
        current_depth = 0.0
        
        for i in range(n_layers):
            # 选择沉积物类型
            sediment_type = random.choice(list(self.sediment_types.keys()))
            type_params = self.sediment_types[sediment_type]
            
            # 生成层参数
            thickness = np.random.uniform(*self.config['sediment']['thickness_range'])
            density = np.random.uniform(*type_params['density'])
            velocity = np.random.uniform(*type_params['velocity'])
            attenuation = np.random.uniform(*type_params['attenuation'])
            
            layer = {
                'layer_id': i,
                'sediment_type': sediment_type,
                'thickness': thickness,
                'density': density,
                'velocity': velocity,
                'attenuation': attenuation,
                'top_depth': current_depth,
                'bottom_depth': current_depth + thickness
            }
            
            layers.append(layer)
            current_depth += thickness
        
        return {
            'layers': layers,
            'total_depth': current_depth,
            'model_id': f"model_{np.random.randint(10000):05d}"
        }
    
    def add_geological_features(self, base_model: Dict) -> Dict:
        """添加地质特征"""
        features = []
        
        # 添加气包特征 (20% 概率)
        if np.random.random() < 0.2:
            gas_layer = self._create_gas_pocket(base_model)
            features.append(gas_layer)
        
        # 添加埋藏物体 (15% 概率)
        if np.random.random() < 0.15:
            buried_object = self._create_buried_object(base_model)
            features.append(buried_object)
        
        # 添加断层 (10% 概率)
        if np.random.random() < 0.1:
            fault = self._create_fault(base_model)
            features.append(fault)
        
        base_model['features'] = features
        return base_model
    
    def _create_gas_pocket(self, model: Dict) -> Dict:
        """创建气包特征"""
        max_depth = model['total_depth'] * 0.8
        depth = np.random.uniform(1.0, max_depth)
        
        return {
            'type': 'gas_pocket',
            'depth': depth,
            'thickness': np.random.uniform(0.1, 0.5),
            'intensity': np.random.uniform(0.3, 0.9)
        }
    
    def _create_buried_object(self, model: Dict) -> Dict:
        """创建埋藏物体"""
        depth = np.random.uniform(0.5, model['total_depth'] * 0.6)
        
        return {
            'type': 'buried_object',
            'depth': depth,
            'object_type': random.choice(['metal', 'rock', 'artifact']),
            'reflectivity': np.random.uniform(0.5, 1.0)
        }
    
    def _create_fault(self, model: Dict) -> Dict:
        """创建断层"""
        return {
            'type': 'fault',
            'position': np.random.uniform(0.3, 0.7) * model['total_depth'],
            'displacement': np.random.uniform(0.1, 1.0)
        }
    
    def generate_reflection_coefficients(self, model: Dict) -> np.ndarray:
        """计算反射系数序列"""
        sample_rate = self.config['simulation']['sample_rate']
        trace_length = self.config['simulation']['trace_length']
        
        # 计算时间轴
        dt = 1.0 / sample_rate
        time_axis = np.arange(trace_length) * dt
        
        # 创建反射系数序列
        reflection_series = np.zeros(trace_length)
        
        for layer in model['layers']:
            # 计算双程走时
            two_way_time = 2 * layer['top_depth'] / layer['velocity']
            sample_index = int(two_way_time / dt)
            
            if sample_index < trace_length:
                # 计算反射系数 (简化公式)
                if layer['layer_id'] == 0:  # 第一层与水界面
                    z_water = 1500 * 1000  # 水声阻抗
                    z_sediment = layer['density'] * layer['velocity']
                    reflection_coeff = (z_sediment - z_water) / (z_sediment + z_water)
                else:
                    # 层间反射系数
                    prev_layer = model['layers'][layer['layer_id'] - 1]
                    z1 = prev_layer['density'] * prev_layer['velocity']
                    z2 = layer['density'] * layer['velocity']
                    reflection_coeff = (z2 - z1) / (z2 + z1)
                
                reflection_series[sample_index] = reflection_coeff
        
        # 添加特征的反射
        for feature in model.get('features', []):
            if feature['type'] in ['gas_pocket', 'buried_object']:
                two_way_time = 2 * feature['depth'] / 1500  # 近似速度
                sample_index = int(two_way_time / dt)
                if sample_index < trace_length:
                    if feature['type'] == 'gas_pocket':
                        reflection_series[sample_index] += feature['intensity'] * 0.5
                    else:
                        reflection_series[sample_index] += feature['reflectivity'] * 0.3
        
        return reflection_series
    
    def generate_segmentation_mask(self, model: Dict) -> np.ndarray:
        """生成语义分割标注掩码"""
        trace_length = self.config['simulation']['trace_length']
        sample_rate = self.config['simulation']['sample_rate']
        dt = 1.0 / sample_rate
        
        # 创建分类掩码 (0: 水, 1: clay, 2: silt, 3: sand, 4: gravel, 5: 特征)
        segmentation_mask = np.zeros(trace_length, dtype=np.int32)
        
        # 为每层分配类别
        type_to_class = {'clay': 1, 'silt': 2, 'sand': 3, 'gravel': 4}
        
        for layer in model['layers']:
            start_sample = int(2 * layer['top_depth'] / layer['velocity'] / dt)
            end_sample = int(2 * layer['bottom_depth'] / layer['velocity'] / dt)
            
            end_sample = min(end_sample, trace_length - 1)
            class_id = type_to_class.get(layer['sediment_type'], 1)
            
            if start_sample < trace_length:
                segmentation_mask[start_sample:end_sample] = class_id
        
        # 标记特征
        for feature in model.get('features', []):
            sample_index = int(2 * feature['depth'] / 1500 / dt)
            if sample_index < trace_length:
                if feature['type'] == 'gas_pocket':
                    segmentation_mask[sample_index] = 5
                elif feature['type'] == 'buried_object':
                    segmentation_mask[sample_index] = 6
        
        return segmentation_mask