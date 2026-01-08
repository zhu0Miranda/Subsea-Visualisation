
import numpy as np
import os
from tqdm import tqdm
import yaml
from typing import Dict, Any

# 硬编码配置，避免文件读取问题
DEFAULT_CONFIG = {
    'simulation': {
        'sample_rate': 100000,
        'duration': 0.1,
        'num_traces': 64,
        'trace_length': 1024
    },
    'source': {
        'frequency': 10000,
        'bandwidth': 8000,
        'pulse_type': 'ricker'
    },
    'sediment': {
        'min_layers': 1,
        'max_layers': 8,
        'thickness_range': [0.1, 5.0],
        'density_range': [1200, 2200],
        'velocity_range': [1450, 2200],
        'attenuation_range': [0.1, 5.0]
    },
    'sediment_types': {
        'clay': {
            'density': [1500, 1700],
            'velocity': [1470, 1550],
            'attenuation': [0.5, 1.5]
        },
        'silt': {
            'density': [1650, 1850],
            'velocity': [1550, 1650],
            'attenuation': [1.0, 2.5]
        },
        'sand': {
            'density': [1800, 2100],
            'velocity': [1650, 1850],
            'attenuation': [2.0, 4.0]
        },
        'gravel': {
            'density': [1900, 2200],
            'velocity': [1800, 2200],
            'attenuation': [3.0, 5.0]
        }
    },
    'noise': {
        'snr_range': [10, 30],
        'reverberation_level': 0.1,
        'electronic_noise': 0.05
    }
}

# 导入自定义模块
from core.sediment_generator import SedimentModelGenerator
from core.acoustic_simulator import AcousticSimulator
from core.sensor_simulator import SensorSimulator
from core.data_augmentation import DataAugmentation
from utils.file_io import DataSaver

class SonarSimulationPipeline:
    """修复的声呐仿真流水线 - 使用硬编码配置"""
    
    def __init__(self):
        # 使用硬编码配置
        self.config = DEFAULT_CONFIG
        
        # 初始化组件
        self.sediment_gen = SedimentModelGenerator()
        self.acoustic_sim = AcousticSimulator(self.config)
        self.sensor_sim = SensorSimulator(self.config)
        self.data_aug = DataAugmentation(self.config)
        self.data_saver = DataSaver("training_data")
        
        # 创建输出目录
        os.makedirs("training_data", exist_ok=True)
    
    def generate_single_sample(self, sample_id: int) -> Dict[str, Any]:
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
    
    def generate_dataset(self, num_samples: int = 100):
        """生成完整数据集"""
        dataset_info = {
            'total_samples': num_samples,
            'creation_date': str(np.datetime64('now'))
        }
        
        for i in tqdm(range(num_samples), desc="生成样本"):
            sample = self.generate_single_sample(i)
            
            # 保存样本
            filename = f"sample_{i:06d}"
            self.data_saver.save_training_sample(
                sample['data'], sample['labels'], sample['metadata'], filename
            )
        
        # 保存数据集信息
        self.data_saver.save_dataset_info(dataset_info)
        print(f"数据集生成完成！共 {num_samples} 个样本")

if __name__ == "__main__":
    # 创建修复的仿真流水线
    pipeline = SonarSimulationPipeline()
    
    # 生成训练数据集
    print("开始生成训练数据集...")
    pipeline.generate_dataset(num_samples=50)  # 先测试50个样本