#-*- coding: GB18030 -*-
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from main import SonarSimulationPipeline

def demo_single_sample():
    """演示生成单个样本 - GB18030版本"""
    try:
        pipeline = SonarSimulationPipeline()
        sample = pipeline.generate_single_sample(0)
        
        # 可视化结果
        data = sample['data']
        labels = sample['labels']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 显示声呐数据
        ax1.plot(data)
        ax1.set_title('Synthetic Sonar Signal')
        ax1.set_xlabel('Sample Points')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        
        # 显示分割标注
        ax2.stem(labels, linefmt='r-', markerfmt='ro', basefmt=' ')
        ax2.set_title('Sediment Layer Classification Labels')
        ax2.set_xlabel('Sample Points')
        ax2.set_ylabel('Class')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('demo_sample_gb18030.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Sample metadata:", sample['metadata'])
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Error in demo: {e}")
        print("Trying alternative approach...")
        run_fallback_demo()

def run_fallback_demo():
    """备用演示方案"""
    print("Running fallback demo without config file...")
    
    # 创建一个简化的演示，不依赖配置文件
    sample_rate = 100000
    duration = 0.1
    trace_length = 1024
    dt = 1.0 / sample_rate
    
    # 生成简单的合成数据
    t = np.arange(0, duration, dt)
    
    # 创建模拟的声呐信号
    # 添加几个峰值模拟层界面
    synthetic_signal = np.zeros(trace_length)
    
    # 在随机位置添加反射峰值
    peak_positions = [100, 300, 600, 800]
    peak_amplitudes = [0.8, 0.6, 0.4, 0.3]
    
    for pos, amp in zip(peak_positions, peak_amplitudes):
        if pos < trace_length:
            synthetic_signal[pos] = amp
    
    # 添加噪声
    noise = np.random.normal(0, 0.05, trace_length)
    synthetic_signal += noise
    
    # 创建对应的标签
    labels = np.zeros(trace_length, dtype=int)
    for i, pos in enumerate(peak_positions):
        if pos < trace_length:
            labels[pos:min(pos+10, trace_length)] = i + 1
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(synthetic_signal)
    ax1.set_title('Fallback Synthetic Sonar Signal')
    ax1.set_xlabel('Sample Points')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    ax2.stem(labels, linefmt='r-', markerfmt='ro', basefmt=' ')
    ax2.set_title('Fallback Sediment Layer Labels')
    ax2.set_xlabel('Sample Points')
    ax2.set_ylabel('Class')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('fallback_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Fallback demo completed!")

if __name__ == "__main__":
    demo_single_sample()