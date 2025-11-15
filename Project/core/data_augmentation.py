#-*- coding: GB18030 -*-
import numpy as np
from scipy import ndimage
from typing import List, Tuple
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

class DataAugmentation:
    """数据增强类"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def apply_augmentation(self, data: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """应用数据增强"""
        augmented_data = data.copy()
        augmented_mask = mask.copy() if mask is not None else None
        
        # 随机选择增强方法
        augmentations = [
            self._time_shift,
            self._amplitude_scaling,
            self._frequency_filter,
            self._random_noise_injection,
            self._spatial_warping
        ]
        
        # 应用1-3种随机增强
        num_augmentations = np.random.randint(1, 4)
        selected_augmentations = np.random.choice(
            augmentations, num_augmentations, replace=False
        )
        
        for aug_func in selected_augmentations:
            if aug_func == self._spatial_warping and augmented_data.ndim >= 2:
                augmented_data, augmented_mask = aug_func(augmented_data, augmented_mask)
            else:
                augmented_data = aug_func(augmented_data)
        
        return augmented_data, augmented_mask
    
    def _time_shift(self, data: np.ndarray) -> np.ndarray:
        """时间偏移"""
        shift = np.random.randint(-10, 10)
        return np.roll(data, shift, axis=-1)
    
    def _amplitude_scaling(self, data: np.ndarray) -> np.ndarray:
        """振幅缩放"""
        scale = np.random.uniform(0.7, 1.3)
        return data * scale
    
    def _frequency_filter(self, data: np.ndarray) -> np.ndarray:
        """频率滤波"""
        # 简单的带通滤波模拟
        if data.ndim == 1:
            fft_data = np.fft.fft(data)
            freq = np.fft.fftfreq(len(data))
            
            # 随机调整频率响应
            low_cut = np.random.uniform(0.1, 0.3)
            high_cut = np.random.uniform(0.6, 0.9)
            
            fft_data[np.abs(freq) < low_cut] *= np.random.uniform(0.5, 1.0)
            fft_data[np.abs(freq) > high_cut] *= np.random.uniform(0.3, 0.8)
            
            return np.real(np.fft.ifft(fft_data))
        return data
    
    def _random_noise_injection(self, data: np.ndarray) -> np.ndarray:
        """随机噪声注入"""
        noise_level = np.random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_level * np.std(data), data.shape)
        return data + noise
    
    def _spatial_warping(self, data: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """空间形变 (对2D/3D数据)"""
        if data.ndim < 2:
            return data, mask
        
        # 应用弹性形变
        displacement = np.random.uniform(-2, 2, data.shape[:2] + (2,))
        
        warped_data = ndimage.map_coordinates(
            data, 
            np.indices(data.shape) + displacement[..., np.newaxis],
            order=1
        )
        
        warped_mask = None
        if mask is not None:
            warped_mask = ndimage.map_coordinates(
                mask.astype(float),
                np.indices(mask.shape) + displacement[..., np.newaxis],
                order=0  # 最近邻插值保持标签完整性
            ).astype(mask.dtype)
        
        return warped_data, warped_mask
    
    def generate_variations(self, base_model: Dict, num_variations: int = 5) -> List[Dict]:
        """生成基础模型的参数变体"""
        variations = []
        
        for i in range(num_variations):
            variation = base_model.copy()
            
            # 对每层参数添加随机扰动
            for layer in variation['layers']:
                layer['density'] *= np.random.uniform(0.9, 1.1)
                layer['velocity'] *= np.random.uniform(0.95, 1.05)
                layer['thickness'] *= np.random.uniform(0.8, 1.2)
            
            variations.append(variation)
        
        return variations