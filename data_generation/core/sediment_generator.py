#-*- coding: GB18030 -*-
"""
Sediment Layer Generator Module
用于生成海底沉积层地质模型和标注数据
"""

import numpy as np
import yaml
import random
from typing import Dict, List, Tuple, Optional, Any

# 尝试导入scipy的ndimage，如果不可用则使用备用方案
try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("警告: scipy.ndimage 不可用，将使用备用滤波方法")


class SedimentModelGenerator:
    """海底沉积层地质模型生成器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化沉积层生成器
        
        参数:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        if config_path:
            # 尝试多种编码读取配置文件
            try:
                with open(config_path, 'r', encoding='gb18030') as f:
                    self.config = yaml.safe_load(f)
                print("SedimentGenerator: 使用GB18030编码读取配置文件")
            except UnicodeDecodeError:
                try:
                    with open(config_path, 'r', encoding='gbk') as f:
                        self.config = yaml.safe_load(f)
                    print("SedimentGenerator: 使用GBK编码读取配置文件")
                except Exception:
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            self.config = yaml.safe_load(f)
                        print("SedimentGenerator: 使用UTF-8编码读取配置文件")
                    except Exception as e:
                        print(f"SedimentGenerator: 无法读取配置文件 {config_path}: {e}")
                        print("使用默认配置")
                        self._set_default_config()
        else:
            # 无配置文件时使用默认配置
            self._set_default_config()
        
        # 获取沉积物类型配置
        self.sediment_types = self.config.get('sediment_types', self._get_default_sediment_types())
        
        # 获取仿真参数
        self.simulation_params = self.config.get('simulation', {
            'sample_rate': 100000,
            'trace_length': 1024
        })
        
        # 获取沉积层参数
        self.sediment_params = self.config.get('sediment', {
            'min_layers': 1,
            'max_layers': 8,
            'thickness_range': [0.1, 5.0]
        })
        
        print(f"SedimentGenerator初始化完成: 采样率={self.simulation_params['sample_rate']}Hz, "
              f"道长度={self.simulation_params['trace_length']}")

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
                'thickness_range': [0.1, 5.0],
                'density_range': [1200, 2200],
                'velocity_range': [1450, 2200],
                'attenuation_range': [0.1, 5.0]
            },
            'sediment_types': self._get_default_sediment_types()
        }

    def _get_default_sediment_types(self) -> Dict[str, Dict]:
        """获取默认的沉积物类型配置"""
        return {
            'clay': {
                'density': [1500, 1700],
                'velocity': [1470, 1550],
                'attenuation': [0.5, 1.5],
                'color': 'blue',
                'description': '粘土 - 细颗粒沉积物'
            },
            'silt': {
                'density': [1650, 1850],
                'velocity': [1550, 1650],
                'attenuation': [1.0, 2.5],
                'color': 'green',
                'description': '粉砂 - 中等颗粒沉积物'
            },
            'sand': {
                'density': [1800, 2100],
                'velocity': [1650, 1850],
                'attenuation': [2.0, 4.0],
                'color': 'yellow',
                'description': '沙 - 粗颗粒沉积物'
            },
            'gravel': {
                'density': [1900, 2200],
                'velocity': [1800, 2200],
                'attenuation': [3.0, 5.0],
                'color': 'red',
                'description': '砾石 - 极粗颗粒沉积物'
            },
            'mixed': {
                'density': [1600, 2000],
                'velocity': [1500, 1800],
                'attenuation': [1.5, 3.5],
                'color': 'purple',
                'description': '混合沉积物'
            }
        }

    def generate_layered_model(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        生成随机层状沉积模型
        
        返回:
            包含沉积层信息的字典
        """
        # 随机确定层数
        min_layers = self.sediment_params['min_layers']
        max_layers = self.sediment_params['max_layers']
        n_layers = random.randint(min_layers, max_layers)
        
        layers = []
        current_depth = 0.0
        
        # 生成每层的参数
        for i in range(n_layers):
            # 随机选择沉积物类型
            available_types = list(self.sediment_types.keys())
            sediment_type = random.choice(available_types)
            type_params = self.sediment_types[sediment_type]
            
            # 生成层参数
            thickness = np.random.uniform(*self.sediment_params['thickness_range'])
            density = np.random.uniform(*type_params['density'])
            velocity = np.random.uniform(*type_params['velocity'])
            attenuation = np.random.uniform(*type_params['attenuation'])
            
            # 创建层信息字典
            layer = {
                'layer_id': i,
                'sediment_type': sediment_type,
                'thickness': round(thickness, 3),
                'density': round(density, 1),
                'velocity': round(velocity, 1),
                'attenuation': round(attenuation, 3),
                'top_depth': round(current_depth, 3),
                'bottom_depth': round(current_depth + thickness, 3),
                'color': type_params.get('color', 'gray'),
                'description': type_params.get('description', '')
            }
            
            layers.append(layer)
            current_depth += thickness
        
        # 生成模型ID
        if model_id is None:
            model_id = f"model_{np.random.randint(10000, 99999)}"
        
        return {
            'model_id': model_id,
            'layers': layers,
            'total_depth': round(current_depth, 3),
            'num_layers': n_layers,
            'water_depth': 0.0,  # 默认为0，表示从海面开始
            'created_time': np.datetime64('now')
        }

    def add_geological_features(self, base_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        添加地质特征到基础模型
        
        参数:
            base_model: 基础沉积层模型
            
        返回:
            添加了地质特征的模型
        """
        features = []
        
        # 添加气包特征 (20% 概率)
        if np.random.random() < 0.2:
            gas_pocket = self._create_gas_pocket(base_model)
            features.append(gas_pocket)
        
        # 添加埋藏物体 (15% 概率)
        if np.random.random() < 0.15:
            buried_object = self._create_buried_object(base_model)
            features.append(buried_object)
        
        # 添加断层 (10% 概率)
        if np.random.random() < 0.1:
            fault = self._create_fault(base_model)
            features.append(fault)
        
        # 添加生物扰动 (25% 概率)
        if np.random.random() < 0.25:
            bioturbation = self._create_bioturbation(base_model)
            features.append(bioturbation)
        
        # 添加不规则层界面 (30% 概率)
        if np.random.random() < 0.3:
            irregular_interface = self._create_irregular_interface(base_model)
            features.append(irregular_interface)
        
        base_model['features'] = features
        return base_model

    def _create_gas_pocket(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """创建气包特征"""
        max_depth = model['total_depth'] * 0.7  # 气包通常不会太深
        depth = np.random.uniform(0.5, max_depth)
        
        return {
            'type': 'gas_pocket',
            'depth': round(depth, 3),
            'thickness': round(np.random.uniform(0.05, 0.3), 3),
            'intensity': round(np.random.uniform(0.3, 0.9), 2),
            'shape': random.choice(['spherical', 'elliptical', 'irregular']),
            'size': round(np.random.uniform(0.1, 1.0), 2),
            'description': '含气沉积物，声阻抗低，反射系数高'
        }

    def _create_buried_object(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """创建埋藏物体"""
        depth = np.random.uniform(0.3, model['total_depth'] * 0.6)
        object_types = ['metal', 'rock', 'artifact', 'wood', 'concrete']
        
        return {
            'type': 'buried_object',
            'object_type': random.choice(object_types),
            'depth': round(depth, 3),
            'size': round(np.random.uniform(0.2, 2.0), 2),
            'reflectivity': round(np.random.uniform(0.5, 1.0), 2),
            'shape': random.choice(['spherical', 'cylindrical', 'rectangular', 'irregular']),
            'orientation': random.choice(['horizontal', 'vertical', 'diagonal']),
            'description': f'埋藏{random.choice(object_types)}物体，反射系数高'
        }

    def _create_fault(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """创建断层"""
        position = np.random.uniform(0.3, 0.7) * model['total_depth']
        fault_types = ['normal', 'reverse', 'strike-slip', 'thrust']
        
        return {
            'type': 'fault',
            'fault_type': random.choice(fault_types),
            'position': round(position, 3),
            'displacement': round(np.random.uniform(0.1, 1.0), 3),
            'dip_angle': round(np.random.uniform(30, 80), 1),
            'length': round(np.random.uniform(1.0, 10.0), 2),
            'description': f'{random.choice(fault_types)}断层，导致地层错断'
        }

    def _create_bioturbation(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """创建生物扰动特征"""
        depth = np.random.uniform(0.1, model['total_depth'] * 0.4)
        
        return {
            'type': 'bioturbation',
            'depth': round(depth, 3),
            'intensity': round(np.random.uniform(0.2, 0.8), 2),
            'thickness': round(np.random.uniform(0.1, 0.5), 3),
            'organism_type': random.choice(['worm', 'crab', 'burrowing_fish', 'unknown']),
            'description': '生物扰动导致的沉积物结构破坏'
        }

    def _create_irregular_interface(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """创建不规则层界面"""
        layer_idx = random.randint(0, len(model['layers']) - 1)
        layer = model['layers'][layer_idx]
        
        return {
            'type': 'irregular_interface',
            'layer_index': layer_idx,
            'interface_depth': round(layer['top_depth'], 3),
            'roughness': round(np.random.uniform(0.05, 0.3), 3),
            'wavelength': round(np.random.uniform(0.5, 5.0), 2),
            'amplitude': round(np.random.uniform(0.05, 0.2), 3),
            'description': '不规则层界面，由水动力作用形成'
        }

    def generate_reflection_coefficients(self, model: Dict[str, Any]) -> np.ndarray:
        """
        计算反射系数序列
        
        参数:
            model: 沉积层模型
            
        返回:
            反射系数序列 (1D numpy数组)
        """
        sample_rate = self.simulation_params['sample_rate']
        trace_length = self.simulation_params['trace_length']
        
        # 计算时间间隔
        dt = 1.0 / sample_rate
        
        # 初始化反射系数序列
        reflection_series = np.zeros(trace_length)
        
        # 计算水-沉积物界面反射系数
        if model['layers']:
            first_layer = model['layers'][0]
            z_water = 1500 * 1000  # 水的声阻抗 (密度 * 速度)
            z_sediment = first_layer['density'] * first_layer['velocity']
            reflection_coeff = (z_sediment - z_water) / (z_sediment + z_water)
            
            # 计算到达时间并设置反射系数
            two_way_time = 2 * first_layer['top_depth'] / first_layer['velocity']
            sample_index = int(two_way_time / dt)
            
            if 0 <= sample_index < trace_length:
                reflection_series[sample_index] = reflection_coeff
        
        # 计算层间反射系数
        for i in range(1, len(model['layers'])):
            prev_layer = model['layers'][i-1]
            curr_layer = model['layers'][i]
            
            z1 = prev_layer['density'] * prev_layer['velocity']
            z2 = curr_layer['density'] * curr_layer['velocity']
            reflection_coeff = (z2 - z1) / (z2 + z1)
            
            # 考虑衰减对反射系数的影响
            attenuation_factor = np.exp(-prev_layer['attenuation'] * prev_layer['thickness'] / 100)
            reflection_coeff *= attenuation_factor
            
            # 计算到达时间
            two_way_time = 2 * curr_layer['top_depth'] / curr_layer['velocity']
            sample_index = int(two_way_time / dt)
            
            if 0 <= sample_index < trace_length:
                reflection_series[sample_index] = reflection_coeff
        
        # 添加地质特征的反射
        for feature in model.get('features', []):
            if feature['type'] in ['gas_pocket', 'buried_object']:
                # 计算特征的双程走时 (使用平均声速1500 m/s)
                two_way_time = 2 * feature['depth'] / 1500
                sample_index = int(two_way_time / dt)
                
                if 0 <= sample_index < trace_length:
                    if feature['type'] == 'gas_pocket':
                        # 气包通常有强负反射
                        reflection_strength = -feature['intensity'] * 0.8
                    else:
                        # 埋藏物体有强正反射
                        reflection_strength = feature['reflectivity'] * 0.6
                    
                    # 添加到反射系数序列
                    reflection_series[sample_index] += reflection_strength
        
        # 添加随机噪声模拟测量误差
        noise_level = 0.01
        reflection_series += np.random.normal(0, noise_level, trace_length) * np.std(reflection_series)
        
        return reflection_series

    def generate_segmentation_mask(self, model: Dict[str, Any]) -> np.ndarray:
        """
        生成语义分割标注掩码
        
        参数:
            model: 沉积层模型
            
        返回:
            分割掩码 (1D numpy数组，整数类型)
        """
        trace_length = self.simulation_params['trace_length']
        sample_rate = self.simulation_params['sample_rate']
        dt = 1.0 / sample_rate
        
        # 类别定义
        # 0: 海水, 1: clay, 2: silt, 3: sand, 4: gravel, 5: mixed
        # 6: gas_pocket, 7: buried_object, 8: fault, 9: bioturbation, 10: irregular_interface
        segmentation_mask = np.zeros(trace_length, dtype=np.int32)
        
        # 为每层沉积物分配类别
        type_to_class = {
            'clay': 1,
            'silt': 2,
            'sand': 3,
            'gravel': 4,
            'mixed': 5
        }
        
        for layer in model['layers']:
            # 计算该层在时间序列中的位置
            start_time = 2 * layer['top_depth'] / layer['velocity']
            end_time = 2 * layer['bottom_depth'] / layer['velocity']
            
            start_sample = int(start_time / dt)
            end_sample = int(end_time / dt)
            
            # 确保索引在有效范围内
            start_sample = max(0, start_sample)
            end_sample = min(trace_length - 1, end_sample)
            
            if start_sample < trace_length:
                class_id = type_to_class.get(layer['sediment_type'], 1)
                segmentation_mask[start_sample:end_sample] = class_id
        
        # 标记地质特征
        for feature in model.get('features', []):
            # 检查特征是否有深度信息
            if 'depth' in feature:
                feature_time = 2 * feature['depth'] / 1500  # 使用平均速度
                sample_index = int(feature_time / dt)
                
                if 0 <= sample_index < trace_length:
                    # 根据特征类型分配类别
                    if feature['type'] == 'gas_pocket':
                        # 气包通常有一定厚度
                        thickness_samples = int(0.1 / dt)  # 假设0.1m厚度
                        start_idx = max(0, sample_index - thickness_samples//2)
                        end_idx = min(trace_length, sample_index + thickness_samples//2)
                        segmentation_mask[start_idx:end_idx] = 6
                        
                    elif feature['type'] == 'buried_object':
                        # 埋藏物体
                        thickness_samples = int(0.2 / dt)  # 假设0.2m厚度
                        start_idx = max(0, sample_index - thickness_samples//2)
                        end_idx = min(trace_length, sample_index + thickness_samples//2)
                        segmentation_mask[start_idx:end_idx] = 7
                        
                    elif feature['type'] == 'fault':
                        # 断层通常影响一个区域
                        thickness_samples = int(0.3 / dt)
                        start_idx = max(0, sample_index - thickness_samples//2)
                        end_idx = min(trace_length, sample_index + thickness_samples//2)
                        segmentation_mask[start_idx:end_idx] = 8
                        
                    elif feature['type'] == 'bioturbation':
                        # 生物扰动区域
                        thickness_samples = int(feature.get('thickness', 0.3) / dt)
                        start_idx = max(0, sample_index - thickness_samples//2)
                        end_idx = min(trace_length, sample_index + thickness_samples//2)
                        segmentation_mask[start_idx:end_idx] = 9
            elif feature['type'] == 'irregular_interface' and 'interface_depth' in feature:
                # 处理不规则层界面
                interface_time = 2 * feature['interface_depth'] / 1500
                sample_index = int(interface_time / dt)
                
                if 0 <= sample_index < trace_length:
                    # 不规则层界面标记为类别10
                    thickness_samples = int(0.05 / dt)  # 薄层
                    start_idx = max(0, sample_index - thickness_samples//2)
                    end_idx = min(trace_length, sample_index + thickness_samples//2)
                    segmentation_mask[start_idx:end_idx] = 10
        
        return segmentation_mask

    def _simple_gaussian_filter(self, data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        简单的Gaussian滤波器实现
        
        参数:
            data: 输入数据
            sigma: 高斯核的标准差
            
        返回:
            滤波后的数据
        """
        # 创建高斯核
        kernel_size = int(4 * sigma + 0.5)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        x = np.arange(-kernel_size//2, kernel_size//2 + 1)
        kernel = np.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        # 应用卷积
        filtered = np.convolve(data, kernel, mode='same')
        return filtered

    def generate_2d_model_variation(self, base_model: Dict[str, Any], num_traces: int = 64) -> np.ndarray:
        """
        生成2D模型变化，用于3D剖面数据生成
        
        参数:
            base_model: 基础模型
            num_traces: 道数
            
        返回:
            2D反射系数变化 (num_traces x trace_length)
        """
        trace_length = self.simulation_params['trace_length']
        reflection_series_2d = np.zeros((num_traces, trace_length))
        
        # 生成基础反射系数序列
        base_reflection = self.generate_reflection_coefficients(base_model)
        
        for i in range(num_traces):
            # 复制基础反射系数
            variation = base_reflection.copy()
            
            # 添加空间变化
            if np.random.random() < 0.3:  # 30%的概率添加局部变化
                # 随机扰动
                random_perturbation = np.random.normal(0, 0.1, trace_length)
                
                # 使用可用的滤波方法
                if HAS_SCIPY:
                    # 使用scipy的gaussian_filter1d
                    variation += ndimage.gaussian_filter1d(random_perturbation, sigma=10)
                else:
                    # 使用自定义的简单高斯滤波器
                    variation += self._simple_gaussian_filter(random_perturbation, sigma=10)
            
            # 模拟横向连续性
            if i > 0:
                # 与前一道保持一定的连续性
                continuity = np.random.uniform(0.7, 0.95)
                variation = continuity * reflection_series_2d[i-1, :] + (1-continuity) * variation
            
            reflection_series_2d[i, :] = variation
        
        return reflection_series_2d

    def get_model_statistics(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取模型统计信息
        
        参数:
            model: 沉积层模型
            
        返回:
            统计信息字典
        """
        layers = model['layers']
        
        # 计算各类型沉积物的厚度
        type_thickness = {}
        for layer in layers:
            sed_type = layer['sediment_type']
            thickness = layer['thickness']
            if sed_type not in type_thickness:
                type_thickness[sed_type] = 0
            type_thickness[sed_type] += thickness
        
        # 计算平均参数
        avg_density = np.mean([layer['density'] for layer in layers])
        avg_velocity = np.mean([layer['velocity'] for layer in layers])
        avg_attenuation = np.mean([layer['attenuation'] for layer in layers])
        
        # 统计特征
        feature_counts = {}
        for feature in model.get('features', []):
            ftype = feature['type']
            if ftype not in feature_counts:
                feature_counts[ftype] = 0
            feature_counts[ftype] += 1
        
        return {
            'model_id': model['model_id'],
            'total_depth': model['total_depth'],
            'num_layers': len(layers),
            'type_distribution': type_thickness,
            'average_density': round(avg_density, 2),
            'average_velocity': round(avg_velocity, 2),
            'average_attenuation': round(avg_attenuation, 3),
            'feature_counts': feature_counts,
            'water_depth': model.get('water_depth', 0.0)
        }

    def print_model_summary(self, model: Dict[str, Any]):
        """打印模型摘要信息"""
        stats = self.get_model_statistics(model)
        
        print("\n" + "="*60)
        print("沉积层模型摘要")
        print("="*60)
        print(f"模型ID: {stats['model_id']}")
        print(f"总深度: {stats['total_depth']} m")
        print(f"层数: {stats['num_layers']}")
        print(f"水深: {stats['water_depth']} m")
        
        print("\n沉积层详细信息:")
        print("-"*60)
        for i, layer in enumerate(model['layers']):
            print(f"层 {i+1}: {layer['sediment_type']}")
            print(f"  深度范围: {layer['top_depth']:.2f} - {layer['bottom_depth']:.2f} m")
            print(f"  厚度: {layer['thickness']:.3f} m")
            print(f"  密度: {layer['density']:.1f} kg/m³")
            print(f"  声速: {layer['velocity']:.1f} m/s")
            print(f"  衰减: {layer['attenuation']:.3f} dB/m/kHz")
        
        if model.get('features'):
            print("\n地质特征:")
            print("-"*60)
            for feature in model['features']:
                print(f"类型: {feature['type']}")
                if 'depth' in feature:
                    print(f"  深度: {feature['depth']:.3f} m")
                if 'description' in feature:
                    print(f"  描述: {feature['description']}")
        
        print("\n统计信息:")
        print("-"*60)
        print(f"平均密度: {stats['average_density']:.1f} kg/m³")
        print(f"平均声速: {stats['average_velocity']:.1f} m/s")
        print(f"平均衰减: {stats['average_attenuation']:.3f} dB/m/kHz")
        
        if stats['feature_counts']:
            print("特征统计:")
            for ftype, count in stats['feature_counts'].items():
                print(f"  {ftype}: {count}个")
        
        print("="*60)


# 测试代码
if __name__ == "__main__":
    print("测试SedimentModelGenerator...")
    
    # 检查scipy是否可用
    if not HAS_SCIPY:
        print("注意: scipy不可用，部分功能将使用简化版本")
    
    # 创建生成器实例
    generator = SedimentModelGenerator()
    
    # 生成沉积层模型
    print("\n1. 生成沉积层模型...")
    base_model = generator.generate_layered_model()
    
    # 添加地质特征
    print("\n2. 添加地质特征...")
    model_with_features = generator.add_geological_features(base_model)
    
    # 生成反射系数序列
    print("\n3. 生成反射系数序列...")
    reflection_series = generator.generate_reflection_coefficients(model_with_features)
    print(f"反射系数序列长度: {len(reflection_series)}")
    print(f"反射系数范围: {reflection_series.min():.4f} 到 {reflection_series.max():.4f}")
    
    # 生成分割掩码
    print("\n4. 生成分割掩码...")
    segmentation_mask = generator.generate_segmentation_mask(model_with_features)
    print(f"分割掩码长度: {len(segmentation_mask)}")
    unique_classes = np.unique(segmentation_mask)
    print(f"包含的类别: {unique_classes}")
    
    # 测试2D模型变化生成
    print("\n5. 测试2D模型变化生成...")
    try:
        reflection_series_2d = generator.generate_2d_model_variation(model_with_features, num_traces=5)
        print(f"2D反射系数矩阵形状: {reflection_series_2d.shape}")
    except Exception as e:
        print(f"2D模型生成失败: {e}")
        print("这通常是因为scipy不可用，但其他功能仍然正常")
    
    # 打印模型摘要
    generator.print_model_summary(model_with_features)
    
    # 获取统计信息
    stats = generator.get_model_statistics(model_with_features)
    print("\n模型统计信息已生成")
    
    print("\n测试完成!")