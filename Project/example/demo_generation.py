#-*- coding: GB18030 -*-
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from main import SonarSimulationPipeline

def demo_single_sample():
    """演示生成单个样本"""
    pipeline = SonarSimulationPipeline()
    sample = pipeline.generate_single_sample(0)
    
    # 可视化结果
    data = sample['data']
    labels = sample['labels']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 显示声呐数据
    ax1.plot(data)
    ax1.set_title('合成声呐信号')
    ax1.set_xlabel('采样点')
    ax1.set_ylabel('振幅')
    ax1.grid(True)
    
    # 显示分割标注
    ax2.stem(labels, linefmt='r-', markerfmt='ro', basefmt=' ')
    ax2.set_title('沉积层分类标注')
    ax2.set_xlabel('采样点')
    ax2.set_ylabel('类别')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('demo_sample.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("样本元数据:", sample['metadata'])

if __name__ == "__main__":
    demo_single_sample()