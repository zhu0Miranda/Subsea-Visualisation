#-*- coding: GB18030 -*-
import numpy as np
import os
from tqdm import tqdm
import yaml
from typing import Dict, Any

# 导入自定义模块
from core.sediment_generator import SedimentModelGenerator
from core.acoustic_simulator import AcousticSimulator
from core.sensor_simulator import SensorSimulator
from core.data_augmentation import DataAugmentation
from utils.file_io import DataSaver

class SonarSimulationPipeline:
    """声呐仿真流水线"""
    
    def __init__(self, config_path: str = "config/simulation_params.yaml"):
        # 加载配置 - 使用GB18030编码
        try:
            with open(config_path, 'r', encoding='gb18030') as f:
                self.config = yaml.safe_load(f)
            print("成功使用GB18030编码读取配置文件")
        except Exception as e:
            print(f"使用GB18030编码读取失败: {e}")
            # 尝试其他编码
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                print("成功使用UTF-8编码读取配置文件")
            except Exception as e2:
                print(f"使用UTF-8编码读取也失败: {e2}")
                raise
        
        # 初始化组件
        self.sediment_gen = SedimentModelGenerator(config_path)
        self.acoustic_sim = AcousticSimulator(self.config)
        self.sensor_sim = SensorSimulator(self.config)
        self.data_aug = DataAugmentation(self.config)
        self.data_saver = DataSaver("training_data")
        
        # 创建输出目录
        os.makedirs("training_data", exist_ok=True)
    
    def generate_single_sample(self, sample_id: int) -> Dict:
        """生成单个训练样本"""
        # 1. 生成地质模型
        base_model = self.sediment_gen.generate_layered_model()
        model_with_features = self.sediment_gen.add_geological_features(base_model)
        
        # 2. 生成反射系数序列
        reflection_series = self.sediment_gen.generate_reflection_coefficients(model_with_features)
        
        # 3. 生成声学响应
        clean_trace = self.acoustic_sim.convolutional_model(reflection_series)
        
        # 4. 添加传感器效应和噪声
        noisy_trace = self.sensor_sim.add_noise(clean_trace)
        
        # 5. 生成标注
        segmentation_mask = self.sediment_gen.generate_segmentation_mask(model_with_features)
        
        # 6. 数据增强
        augmented_trace, augmented_mask = self.data_aug.apply_augmentation(
            noisy_trace, segmentation_mask
        )
        
        # 准备元数据
        metadata = {
            'sample_id': sample_id,
            'model_id': model_with_features['model_id'],
            'num_layers': len(model_with_features['layers']),
            'total_depth': model_with_features['total_depth'],
            'layer_info': model_with_features['layers'],
            'features': model_with_features.get('features', [])
        }
        
        return {
            'data': augmented_trace,
            'labels': augmented_mask,
            'metadata': metadata
        }
    
    def generate_3d_sample(self, sample_id: int) -> Dict:
        """生成3D训练样本"""
        # 生成基础地质模型
        base_model = self.sediment_gen.generate_layered_model()
        
        # 创建2D反射系数变化
        num_traces = self.config['simulation']['num_traces']
        trace_length = self.config['simulation']['trace_length']
        
        reflection_series_2d = np.zeros((num_traces, num_traces, trace_length))
        
        # 为每个位置生成略微不同的反射序列
        base_reflection = self.sediment_gen.generate_reflection_coefficients(base_model)
        
        for i in range(num_traces):
            for j in range(num_traces):
                # 添加空间变化
                variation = base_reflection.copy()
                # 随机扰动
                if np.random.random() < 0.3:  # 30%的概率添加局部变化
                    variation += np.random.normal(0, 0.1, trace_length)
                reflection_series_2d[i, j, :] = variation
        
        # 生成3D声学响应
        profile_3d = self.acoustic_sim.generate_3d_profile(reflection_series_2d)
        
        # 添加波束模式和噪声
        profile_3d = self.sensor_sim.apply_beam_pattern(profile_3d)
        profile_3d = self.sensor_sim.add_noise(profile_3d)
        
        # 生成标注 (使用基础模型的标注)
        segmentation_mask = self.sediment_gen.generate_segmentation_mask(base_model)
        
        metadata = {
            'sample_id': sample_id,
            'model_id': base_model['model_id'],
            'dimensions': profile_3d.shape,
            'is_3d': True
        }
        
        return {
            'data': profile_3d,
            'labels': segmentation_mask,
            'metadata': metadata
        }
    
    def generate_dataset(self, num_samples: int = 1000, use_3d: bool = False):
        """生成完整数据集"""
        dataset_info = {
            'total_samples': num_samples,
            '3d_samples': 0,
            '1d_samples': 0,
            'sediment_type_distribution': {},
            'creation_date': str(np.datetime64('now'))
        }
        
        for i in tqdm(range(num_samples)):
            if use_3d and (i % 5 == 0):  # 每5个样本生成一个3D样本
                sample = self.generate_3d_sample(i)
                dataset_info['3d_samples'] += 1
            else:
                sample = self.generate_single_sample(i)
                dataset_info['1d_samples'] += 1
            
            # 保存样本
            filename = f"sample_{i:06d}"
            self.data_saver.save_training_sample(
                sample['data'], sample['labels'], sample['metadata'], filename
            )
            
            # 更新统计信息
            self._update_dataset_stats(dataset_info, sample)
        
        # 保存数据集信息
        self.data_saver.save_dataset_info(dataset_info)
        print(f"数据集生成完成！共 {num_samples} 个样本")
    
    def _update_dataset_stats(self, dataset_info: Dict, sample: Dict):
        """更新数据集统计信息"""
        metadata = sample['metadata']
        for layer in metadata.get('layer_info', []):
            sediment_type = layer['sediment_type']
            if sediment_type not in dataset_info['sediment_type_distribution']:
                dataset_info['sediment_type_distribution'][sediment_type] = 0
            dataset_info['sediment_type_distribution'][sediment_type] += 1

if __name__ == "__main__":
    # 创建仿真流水线
    pipeline = SonarSimulationPipeline()
    
    # 生成训练数据集
    print("开始生成训练数据集...")
    pipeline.generate_dataset(num_samples=1000, use_3d=True)