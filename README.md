## 📖 项目概述

这是一个用于生成三维浅剖声呐仿真数据的综合系统，专门为神经网络训练提供高质量的带标注海底沉积层数据。项目结合了声学物理仿真、地质建模和机器学习数据增强技术，能够高效生成用于海底沉积层识别与分类的仿真训练数据集。

## 📂 项目结构

```
sonar_simulation/
├── config/                    # 配置文件目录
│   └── simulation_params.yaml # 仿真参数配置文件
├── core/                      # 核心仿真模块
│   ├── sediment_generator.py  # 沉积层地质模型生成器
│   ├── acoustic_simulator.py  # 声学响应模拟器
│   ├── sensor_simulator.py    # 传感器效应模拟器
│   └── data_augmentation.py   # 数据增强模块
├── utils/                     # 工具模块
│   ├── file_io.py            # 数据存储和读取工具
│   └── encoding_helper.py     # 编码处理工具
├── examples/                  # 示例程序
│   └── simple_demo.py        # 简化演示版本
├── main.py                   # 主仿真程序
├── run_project.py            # 项目启动脚本
└── requirements.txt          # 依赖包列表
```

## 🔧 核心功能模块

### 1. 沉积层地质模型生成器
- 随机生成多层层状沉积结构
- 支持粘土、粉砂、沙、砾石等沉积物类型
- 添加地质特征：气包、埋藏物体、断层
- 自动生成语义分割标注

### 2. 声学响应模拟器
- 生成Ricker子波和线性调频脉冲
- 基于卷积模型模拟声波传播
- 支持1D、2D、3D数据生成
- 模拟地层衰减效应

### 3. 传感器效应模拟器
- 添加多种噪声：高斯噪声、混响噪声、脉冲噪声
- 模拟波束方向性模式
- 设置可调信噪比

### 4. 数据增强模块
- 时间偏移、振幅缩放
- 频率滤波、随机噪声注入
- 空间形变增强
- 参数变体生成

## 🚀 快速开始

### 环境要求
- Python 3.8+
- Windows/Linux/macOS

### 安装依赖
```bash
pip install -r requirements.txt
```

或者手动安装：
```bash
pip install numpy scipy matplotlib pyyaml h5py tqdm chardet
```

### 运行演示
1. **简单演示（推荐首次运行）**：
```bash
python examples/simple_demo.py
```

2. **完整演示**：
```bash
python run_project.py
```

3. **生成训练数据集**：
```bash
python main.py
```
