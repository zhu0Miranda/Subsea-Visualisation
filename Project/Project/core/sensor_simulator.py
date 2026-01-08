#-*- coding: GB18030 -*-
import numpy as np
from scipy import ndimage
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

class SensorSimulator:
    """传感器效应模拟器"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def add_noise(self, clean_data: np.ndarray) -> np.ndarray:
        """添加各类噪声"""
        snr_db = np.random.uniform(*self.config['noise']['snr_range'])
        snr_linear = 10 ** (snr_db / 10)
        
        # 计算信号功率
        signal_power = np.mean(clean_data ** 2)
        noise_power = signal_power / snr_linear
        
        # 添加高斯白噪声
        gaussian_noise = np.random.normal(0, np.sqrt(noise_power), clean_data.shape)
        
        # 添加混响噪声
        reverberation = self._generate_reverberation(clean_data.shape)
        
        # 添加脉冲噪声 (模拟生物噪声)
        impulsive_noise = self._generate_impulsive_noise(clean_data.shape)
        
        noisy_data = clean_data + gaussian_noise + \
                    reverberation * self.config['noise']['reverberation_level'] + \
                    impulsive_noise
        
        return noisy_data
    
    def _generate_reverberation(self, shape: tuple) -> np.ndarray:
        """生成混响噪声"""
        # 使用自回归模型模拟混响
        reverberation = np.zeros(shape)
        if len(shape) == 1:
            # 1D 混响
            for i in range(1, len(reverberation)):
                reverberation[i] = 0.7 * reverberation[i-1] + 0.3 * np.random.normal(0, 1)
        else:
            # 2D/3D 混响
            reverberation = np.random.randn(*shape)
            # 应用空间平滑模拟相关混响
            reverberation = ndimage.gaussian_filter(reverberation, sigma=2.0)
        
        return reverberation / np.max(np.abs(reverberation))
    
    def _generate_impulsive_noise(self, shape: tuple) -> np.ndarray:
        """生成脉冲噪声"""
        impulsive_noise = np.zeros(shape)
        num_impulses = max(1, int(0.01 * impulsive_noise.size))  # 1% 的脉冲
        
        if len(shape) == 1:
            positions = np.random.randint(0, len(impulsive_noise), num_impulses)
            impulsive_noise[positions] = np.random.uniform(0.5, 2.0, num_impulses)
        else:
            # 对于多维数据，随机选择位置添加脉冲
            coords = [np.random.randint(0, dim, num_impulses) for dim in shape]
            impulsive_noise[tuple(coords)] = np.random.uniform(0.5, 2.0, num_impulses)
        
        return impulsive_noise
    
    def apply_beam_pattern(self, data_3d: np.ndarray) -> np.ndarray:
        """应用波束方向性模式"""
        if data_3d.ndim != 3:
            return data_3d
        
        num_traces = data_3d.shape[0]
        center = num_traces // 2
        
        # 创建高斯波束模式
        x, y = np.meshgrid(np.arange(num_traces), np.arange(num_traces))
        distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        beam_pattern = np.exp(-(distance ** 2) / (2 * (center / 2) ** 2))
        
        # 应用波束模式到每个时间切片
        beamformed_data = np.zeros_like(data_3d)
        for t in range(data_3d.shape[2]):
            beamformed_data[:, :, t] = data_3d[:, :, t] * beam_pattern
        
        return beamformed_data