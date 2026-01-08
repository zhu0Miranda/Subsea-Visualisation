#-*- coding: GB18030 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class SimpleSonarDemo:
    """简化的声呐演示，不依赖配置文件"""
    
    def __init__(self):
        # 硬编码配置参数
        self.sample_rate = 100000  # Hz
        self.duration = 0.1        # s
        self.trace_length = 1024
        self.dt = 1.0 / self.sample_rate
        
    def generate_source_wavelet(self, freq=10000):
        """生成声源子波"""
        t = np.arange(0, self.duration, self.dt)
        # Ricker 子波
        t0 = 1.0 / freq
        wavelet = (1 - 2 * (np.pi * freq * (t - t0)) ** 2) * \
                 np.exp(-(np.pi * freq * (t - t0)) ** 2)
        return wavelet / np.max(np.abs(wavelet))
    
    def generate_sediment_model(self):
        """生成简单的沉积层模型"""
        # 创建3层沉积模型
        layers = [
            {'type': 'sand', 'depth': 0.5, 'velocity': 1800, 'density': 2000},
            {'type': 'silt', 'depth': 1.5, 'velocity': 1600, 'density': 1800},
            {'type': 'clay', 'depth': 3.0, 'velocity': 1500, 'density': 1700}
        ]
        return layers
    
    def generate_reflection_coefficients(self, layers):
        """生成反射系数序列"""
        reflection_series = np.zeros(self.trace_length)
        
        for i, layer in enumerate(layers):
            two_way_time = 2 * layer['depth'] / layer['velocity']
            sample_index = int(two_way_time / self.dt)
            
            if sample_index < self.trace_length:
                if i == 0:  # 第一层与水界面
                    z_water = 1500 * 1000
                    z_sediment = layer['density'] * layer['velocity']
                    reflection_coeff = (z_sediment - z_water) / (z_sediment + z_water)
                else:
                    # 简化计算
                    reflection_coeff = 0.2
                
                reflection_series[sample_index] = reflection_coeff
        
        return reflection_series
    
    def generate_synthetic_trace(self):
        """生成合成声呐轨迹"""
        # 1. 生成沉积模型
        layers = self.generate_sediment_model()
        
        # 2. 生成反射系数
        reflection_series = self.generate_reflection_coefficients(layers)
        
        # 3. 生成子波
        wavelet = self.generate_source_wavelet()
        
        # 4. 卷积生成合成记录
        synthetic_trace = np.convolve(reflection_series, wavelet, mode='same')
        
        # 5. 添加噪声
        noise = np.random.normal(0, 0.1, len(synthetic_trace))
        synthetic_trace += noise
        
        return synthetic_trace, layers
    
    def generate_segmentation_mask(self, layers):
        """生成分割掩码"""
        segmentation_mask = np.zeros(self.trace_length, dtype=np.int32)
        type_to_class = {'sand': 1, 'silt': 2, 'clay': 3}
        
        for layer in layers:
            sample_index = int(2 * layer['depth'] / layer['velocity'] / self.dt)
            if sample_index < self.trace_length:
                class_id = type_to_class.get(layer['type'], 1)
                # 设置一个区域而不是单个点
                start_idx = max(0, sample_index - 5)
                end_idx = min(self.trace_length, sample_index + 5)
                segmentation_mask[start_idx:end_idx] = class_id
        
        return segmentation_mask

def run_simple_demo():
    """运行简化演示"""
    demo = SimpleSonarDemo()
    
    # 生成数据
    synthetic_trace, layers = demo.generate_synthetic_trace()
    segmentation_mask = demo.generate_segmentation_mask(layers)
    
    # 可视化结果
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # 显示合成声呐信号
    ax1.plot(synthetic_trace)
    ax1.set_title('合成声呐信号')
    ax1.set_xlabel('采样点')
    ax1.set_ylabel('振幅')
    ax1.grid(True)
    
    # 显示反射系数
    reflection_series = demo.generate_reflection_coefficients(layers)
    ax2.stem(reflection_series, linefmt='b-', markerfmt='bo', basefmt=' ')
    ax2.set_title('反射系数序列')
    ax2.set_xlabel('采样点')
    ax2.set_ylabel('反射系数')
    ax2.grid(True)
    
    # 显示分割标注
    ax3.stem(segmentation_mask, linefmt='r-', markerfmt='ro', basefmt=' ')
    ax3.set_title('沉积层分类标注 (1:沙, 2:粉砂, 3:粘土)')
    ax3.set_xlabel('采样点')
    ax3.set_ylabel('类别')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('simple_demo_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("沉积层信息:")
    for i, layer in enumerate(layers):
        print(f"  第{i+1}层: {layer['type']}, 深度: {layer['depth']}m, 速度: {layer['velocity']}m/s")
    
    print(f"\n生成的信号长度: {len(synthetic_trace)} 个采样点")
    print(f"采样率: {demo.sample_rate} Hz")

if __name__ == "__main__":
    run_simple_demo()