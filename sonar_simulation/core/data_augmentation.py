#-*- coding: GB18030 -*-

"""
Data Augmentation Module
用于增强声呐仿真数据，增加数据多样性
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# 尝试导入scipy，如果不可用则使用备用方案
try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy.ndimage not available, using fallback methods")


class DataAugmentation:
    """数据增强类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def apply_augmentation(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        应用数据增强
        
        参数:
            data: 输入数据
            mask: 对应的分割掩码 (可选)
            
        返回:
            (增强后的数据, 增强后的掩码)
        """
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
            # 处理返回类型
            result = aug_func(augmented_data, augmented_mask)
            
            if isinstance(result, tuple):
                # 如果函数返回元组 (data, mask)
                if len(result) == 2:
                    augmented_data, augmented_mask = result
                else:
                    augmented_data = result[0]
            else:
                # 如果函数只返回数据
                augmented_data = result
        
        return augmented_data, augmented_mask
    
    def _time_shift(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """时间偏移"""
        shift = np.random.randint(-10, 10)
        shifted_data = np.roll(data, shift, axis=-1)
        
        shifted_mask = None
        if mask is not None:
            shifted_mask = np.roll(mask, shift, axis=-1)
        
        return shifted_data, shifted_mask
    
    def _amplitude_scaling(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """振幅缩放"""
        scale = np.random.uniform(0.7, 1.3)
        scaled_data = data * scale
        
        return scaled_data, mask
    
    def _frequency_filter(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """频率滤波"""
        if hasattr(data, 'ndim'):  # 确保是numpy数组
            if data.ndim == 1:
                # 简单的带通滤波模拟
                fft_data = np.fft.fft(data)
                freq = np.fft.fftfreq(len(data))
                
                # 随机调整频率响应
                low_cut = np.random.uniform(0.1, 0.3)
                high_cut = np.random.uniform(0.6, 0.9)
                
                # 应用带通滤波
                fft_data[np.abs(freq) < low_cut] *= np.random.uniform(0.5, 1.0)
                fft_data[np.abs(freq) > high_cut] *= np.random.uniform(0.3, 0.8)
                
                filtered_data = np.real(np.fft.ifft(fft_data))
                return filtered_data, mask
        
        return data, mask
    
    def _random_noise_injection(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """随机噪声注入"""
        noise_level = np.random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_level * np.std(data), data.shape)
        noisy_data = data + noise
        
        return noisy_data, mask
    
    def _spatial_warping(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """空间形变 (对2D/3D数据)"""
        if hasattr(data, 'ndim') and data.ndim >= 2 and HAS_SCIPY:
            # 应用弹性形变
            displacement = np.random.uniform(-2, 2, data.shape[:2] + (2,))
            
            warped_data = ndimage.map_coordinates(
                data, 
                np.indices(data.shape) + displacement[..., np.newaxis],
                order=1
            )
            
            warped_mask = mask
            if mask is not None:
                warped_mask = ndimage.map_coordinates(
                    mask.astype(float),
                    np.indices(mask.shape) + displacement[..., np.newaxis],
                    order=0  # 最近邻插值保持标签完整性
                ).astype(mask.dtype)
            
            return warped_data, warped_mask
        
        return data, mask
    
    def generate_variations(self, base_model: Dict[str, Any], num_variations: int = 5) -> List[Dict[str, Any]]:
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


# 测试代码
if __name__ == "__main__":
    # 测试数据增强功能
    print("Testing DataAugmentation module...")
    
    # 创建测试配置
    test_config = {
        'noise': {
            'snr_range': [10, 30]
        }
    }
    
    # 创建增强器
    augmenter = DataAugmentation(test_config)
    
    # 创建测试数据
    test_data = np.sin(np.linspace(0, 10, 1000))
    test_mask = np.zeros(1000, dtype=int)
    test_mask[300:400] = 1
    test_mask[600:700] = 2
    
    print(f"Original data shape: {test_data.shape}")
    print(f"Original mask shape: {test_mask.shape}")
    
    # 测试各种增强
    print("\nTesting augmentations:")
    
    # 测试时间偏移
    shifted_data, shifted_mask = augmenter._time_shift(test_data, test_mask)
    print(f"Time shift: data shape {shifted_data.shape}, mask shape {shifted_mask.shape}")
    
    # 测试振幅缩放
    scaled_data, scaled_mask = augmenter._amplitude_scaling(test_data, test_mask)
    print(f"Amplitude scaling: data range [{scaled_data.min():.3f}, {scaled_data.max():.3f}]")
    
    # 测试频率滤波
    filtered_data, filtered_mask = augmenter._frequency_filter(test_data, test_mask)
    print(f"Frequency filter: applied successfully")
    
    # 测试噪声注入
    noisy_data, noisy_mask = augmenter._random_noise_injection(test_data, test_mask)
    print(f"Noise injection: added noise")
    
    # 测试完整增强流程
    print("\nTesting full augmentation pipeline:")
    augmented_data, augmented_mask = augmenter.apply_augmentation(test_data, test_mask)
    print(f"Augmented data shape: {augmented_data.shape}")
    print(f"Augmented mask shape: {augmented_mask.shape}")
    
    print("\nData augmentation tests completed!")