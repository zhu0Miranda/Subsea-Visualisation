## 📖 sonar_simulation概述

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
└── run_project.py            # 项目启动脚本
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



## 📚 核心模块原理详解

### 1. **`core/sediment_generator.py` - 沉积层地质模型生成器**

#### 🎯 **功能概述**
基于地质统计和沉积学原理，生成随机但物理合理的海底沉积层模型。

#### 🔬 **物理与地质基础**

**1.1 沉积物分类系统**
```python
# 基于Shepard三角图分类法的简化
sediment_types = {
    'clay':    # 粘土 (<0.004mm) - 细颗粒
    'silt':    # 粉砂 (0.004-0.063mm) - 中等颗粒  
    'sand':    # 沙 (0.063-2mm) - 粗颗粒
    'gravel':  # 砾石 (>2mm) - 极粗颗粒
}
```
**原理依据**：Shepard 沉积物分类法，结合声学特性简化。

**1.2 声学参数范围**（基于Hamilton）
```python

'clay': {
    'density': [1500, 1700],      # kg/m³
    'velocity': [1470, 1550],     # m/s
    'attenuation': [0.5, 1.5]     # dB/m/kHz
}
```
**物理意义**：
- **密度**：沉积物颗粒与孔隙水的加权平均
- **声速**：沉积物骨架刚度与孔隙度的函数
- **衰减**：粘滞损失和散射损失的总和

#### ⚙️ **核心算法原理**

**1.3 层状模型生成算法**
```python
def generate_layered_model():
    # 1. 马尔可夫链选择沉积物序列
    #    模拟沉积环境的序列相关性
    #    例如：砂→泥的向上变细序列
    
    # 2. 厚度分布的随机生成
    #    基于对数正态分布（实际沉积特征）
    thickness = exp(normal(μ, σ))
    
    # 3. 物理参数的随机扰动
    #    在类型典型值附近正态分布
    velocity = normal(μ_type, σ_type)
```
**地质原理**：Walther相律（沉积相的垂向序列反映横向相变）

**1.4 反射系数计算**
```python
def calculate_reflection_coefficient(z1, z2):
    """
    基于Zoeppritz方程的法向入射简化
    R = (Z₂ - Z₁) / (Z₂ + Z₁)
    
    其中：Z = ρ·v (声阻抗)
    
    物理意义：
    - R>0: 从低声阻抗到高声阻抗（硬界面）
    - R<0: 从高声阻抗到低声阻抗（软界面）
    - |R|: 反射能量比例
    """
    return (z2 - z1) / (z2 + z1)
```
**理论基础**：Zoeppritz 弹性波反射透射理论

**1.5 时间-深度转换**
```python
# 双程走时计算
two_way_time = 2 * depth / velocity

# 采样点位置计算
sample_index = int(two_way_time / dt)
```
**物理原理**：声波在介质中的传播时间与路径长度成正比

#### 🎲 **随机性设计**
```python
# 设计原则：可控的随机性
1. 层数：均匀分布(1-8层) - 控制复杂度
2. 厚度：对数正态分布 - 自然沉积特征
3. 类型：马尔可夫链 - 序列相关性
4. 特征：概率触发 - 模拟自然出现频率
```

#### 📊 **输出数据结构**
```python
model = {
    'layers': [  # 按深度排序的沉积层
        {
            'sediment_type': 'clay',
            'density': 1600.5,     # kg/m³
            'velocity': 1510.2,    # m/s
            'thickness': 1.23,     # m
            'top_depth': 0.0,      # m
            'bottom_depth': 1.23   # m
        },
        # ... 更多层
    ],
    'features': [  # 地质特征
        {
            'type': 'gas_pocket',
            'depth': 2.5,          # m
            'intensity': 0.7       # 无量纲
        }
    ]
}
```

---

### 2. **`core/acoustic_simulator.py` - 声学响应模拟器**

#### 🎯 **功能概述**
模拟声波在沉积层中的传播，将地质模型转换为声呐信号。

#### 🔬 **物理基础**

**2.1 声波方程与简化**
```python
# 完整声波方程（忽略衰减）
∂²p/∂t² = v² ∇²p

# 一维水平层状介质解
p(z,t) = R(ω) * exp(i(kz - ωt))
```
**简化假设**：
1. 水平层状介质
2. 法向入射平面波
3. 忽略多次反射
4. 忽略横波

**2.2 卷积模型理论基础**
```python
# 线性时不变系统理论
s(t) = r(t) ⊛ w(t) + n(t)

# 其中：
# s(t): 接收信号（输出）
# r(t): 反射系数序列（系统冲激响应）
# w(t): 声源子波（输入）
# n(t): 噪声（干扰）
```
**系统理论**：将地层视为线性滤波器，声源子波为输入

#### ⚙️ **核心算法原理**

**2.3 Ricker子波生成**
```python
def ricker_wavelet(frequency, duration):
    """
    Mexican hat wavelet (Ricker, 1953)
    w(t) = [1 - 2(πfτ)²] exp[-(πfτ)²]
    
    其中：τ = t - t₀, t₀ = 1/f
    
    特性：
    - 零相位：主瓣对称，分辨率高
    - 有限带宽：中心频率f，带宽~f
    - 解析形式：便于计算
    """
    t = np.arange(0, duration, dt)
    t0 = 1.0 / frequency
    tau = t - t0
    return (1 - 2 * (np.pi * frequency * tau)**2) * \
           np.exp(-(np.pi * frequency * tau)**2)
```
**优势**：零相位、解析形式、物理上合理的震源函数

**2.4 卷积运算实现**
```python
def convolutional_model(reflection_series, wavelet):
    """
    一维卷积实现地质滤波
    
    物理过程：
    1. 每个反射界面产生一个反射子波
    2. 所有反射子波叠加（线性叠加原理）
    3. 考虑到达时间延迟
    
    数学实现：
    synthetic[n] = Σ_{k} r[k] * w[n-k]
    """
    return np.convolve(reflection_series, wavelet, mode='same')
```

**2.5 衰减效应模拟**
```python
def apply_attenuation(trace, attenuation_coefficient):
    """
    模拟声波在沉积物中的能量衰减
    
    模型：指数衰减
    A(t) = A₀ * exp(-α * v * t)
    
    其中：
    - α: 衰减系数 (dB/m/kHz → Nepers/m)
    - v: 声速 (m/s)
    - t: 传播时间 (s)
    """
    t = np.arange(len(trace)) * dt
    attenuation_factor = np.exp(-attenuation_coefficient * t)
    return trace * attenuation_factor
```
**物理机制**：粘滞吸收、颗粒摩擦、孔隙流体流动

#### 📈 **参数设置依据**

**2.6 典型参数值**
```python
# 基于典型浅剖声呐系统
source_frequency = 10e3      # 10 kHz - 平衡分辨率与穿透
bandwidth = 8e3             # 8 kHz - 典型相对带宽80%
duration = 0.1              # 100 ms - 足够覆盖几十米深度
```

---

### 3. **`core/sensor_simulator.py` - 传感器效应模拟器**

#### 🎯 **功能概述**
模拟真实声呐系统的测量效应和环境噪声，使仿真数据更接近实测数据。

#### 🔬 **物理与工程基础**

**3.1 噪声类型与物理机制**
```python
noise_types = {
    'gaussian':     # 热噪声、电子噪声 - 加性白噪声
    'reverberation': # 混响 - 多次反射、散射累积
    'impulsive':    # 脉冲噪声 - 生物活动、突发干扰
}
```

**3.2 信噪比模型**
```python
def calculate_snr(signal_power, noise_power):
    """
    信噪比定义
    SNR = 10·log₁₀(P_signal / P_noise) dB
    
    工程意义：
    - SNR>20 dB: 高质量信号
    - SNR~10 dB: 典型工作条件  
    - SNR<5 dB: 信噪比差，需要处理
    """
    return 10 * np.log10(signal_power / noise_power)
```

#### ⚙️ **核心算法原理**

**3.3 高斯白噪声生成**
```python
def add_gaussian_noise(signal, snr_db):
    """
    加性高斯白噪声模型
    
    物理基础：中心极限定理
    - 大量独立噪声源叠加 → 高斯分布
    - 功率谱平坦 → 白噪声
    
    实现步骤：
    1. 计算信号功率 P_signal = E[|s(t)|²]
    2. 根据SNR计算噪声功率 P_noise = P_signal / 10^(SNR/10)
    3. 生成高斯噪声 N(0, σ²), σ = sqrt(P_noise)
    """
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise
```

**3.4 混响噪声模型**
```python
def generate_reverberation(shape):
    """
    自回归模型模拟混响
    
    物理机制：多次反射的累积效应
    
    数学模型：AR(1)过程
    r[n] = α·r[n-1] + (1-α)·ε[n]
    
    其中：
    - α: 衰减系数（0<α<1）
    - ε[n]: 随机激励
    """
    reverberation = np.zeros(shape)
    alpha = 0.7  # 混响衰减率
    for i in range(1, len(reverberation)):
        reverberation[i] = alpha * reverberation[i-1] + \
                          (1-alpha) * np.random.normal(0, 1)
    return reverberation
```

**3.5 波束方向性模拟**
```python
def apply_beam_pattern(data_3d):
    """
    模拟声呐阵列的波束方向性
    
    物理基础：阵列信号处理
    
    简化模型：高斯波束
    B(θ) = exp(-θ²/(2σ²))
    
    其中：
    - θ: 偏离主轴角度
    - σ: 波束宽度参数
    """
    # 创建二维高斯分布模拟波束
    x, y = np.meshgrid(np.arange(num_traces), np.arange(num_traces))
    distance = np.sqrt((x - center)**2 + (y - center)**2)
    beam_pattern = np.exp(-(distance**2) / (2 * (center/2)**2))
    
    # 应用到每个时间切片
    for t in range(data_3d.shape[2]):
        data_3d[:, :, t] *= beam_pattern
    
    return data_3d
```

---

### 4. **`core/data_augmentation.py` - 数据增强器**

#### 🎯 **功能概述**
通过对仿真数据进行随机变换，增加数据多样性，提高神经网络泛化能力。

#### 🔬 **机器学习原理**

**4.1 数据增强的统计意义**
```python
# 通过变换扩展数据分布
原始分布: P(X)
增强后分布: ∫ P(X|T) P(T) dT

# 提高模型鲁棒性
模型学习: f(X) ≈ f(T(X)) 对于合理变换T
```

#### ⚙️ **核心算法原理**

**4.2 时间偏移增强**
```python
def time_shift(data, shift):
    """
    模拟声呐系统的时间同步误差
    
    物理意义：发射-接收时间戳微小偏差
    
    实现：循环移位
    shifted = roll(data, shift)
    
    注意：对应的标注也需要同步移位
    """
    return np.roll(data, shift, axis=-1)
```

**4.3 频率滤波增强**
```python
def frequency_filter(data):
    """
    模拟不同声呐系统的频率响应差异
    
    物理意义：换能器频率响应、水声信道滤波
    
    实现：频域滤波
    1. FFT到频域
    2. 应用随机频率响应
    3. IFFT回时域
    """
    fft_data = np.fft.fft(data)
    freq = np.fft.fftfreq(len(data))
    
    # 随机带通滤波
    low_cut = random(0.1, 0.3)    # 随机低截频率
    high_cut = random(0.6, 0.9)   # 随机高截频率
    
    fft_data[np.abs(freq) < low_cut] *= random(0.5, 1.0)
    fft_data[np.abs(freq) > high_cut] *= random(0.3, 0.8)
    
    return np.real(np.fft.ifft(fft_data))
```

**4.4 振幅缩放增强**
```python
def amplitude_scaling(data, scale):
    """
    模拟声呐增益设置的差异
    
    物理意义：发射功率、接收增益的变化
    
    实现：线性缩放
    scaled = data * scale
    
    注意：保持信号统计特性
    """
    return data * scale
```

**4.5 弹性形变增强**
```python
def spatial_warping(data, mask):
    """
    模拟沉积层的空间形变
    
    地质意义：沉积过程的不均匀性
    
    实现：弹性形变场
    1. 生成随机位移场
    2. 应用样条插值
    3. 保持拓扑结构
    """
    # 生成随机位移场（低频率，保证平滑）
    displacement = random_displacement_field(shape, scale=2.0)
    
    # 应用形变（数据双线性插值，标注最近邻）
    warped_data = map_coordinates(data, displacement, order=1)
    warped_mask = map_coordinates(mask, displacement, order=0)
    
    return warped_data, warped_mask
```


---

## 🎯 物理参数与单位系统

### 基本物理量
| 量 | 符号 | 单位 | 典型值范围 |
|----|------|------|-----------|
| 密度 | ρ | kg/m³ | 1500-2200 |
| 声速 | v | m/s | 1450-2200 |
| 厚度 | d | m | 0.1-5.0 |
| 频率 | f | Hz | 3k-12k |
| 时间 | t | s | 0-0.1 |
| 衰减 | α | dB/m/kHz | 0.1-5.0 |

### 导出量
| 量 | 公式 | 单位 | 物理意义 |
|----|------|------|----------|
| 声阻抗 | Z = ρ·v | kg/(m²·s) | 介质声学硬度 |
| 反射系数 | R = (Z₂-Z₁)/(Z₂+Z₁) | 无量纲 | 界面反射强度 |
| 双程走时 | t = 2d/v | s | 声波往返时间 |
| 采样间隔 | dt = 1/fs | s | 时间分辨率 |

---

## 🔬 算法复杂度分析

### 时间复杂度
```python
# 单个样本生成复杂度
O_total = O_geo + O_acoustic + O_sensor + O_augment

O_geo = O(n_layers)         ~ O(1-8)
O_acoustic = O(N·logN)      # 卷积或FFT，N=1024
O_sensor = O(N)             # 噪声添加
O_augment = O(N·logN)       # 可能包含FFT

# 总复杂度：~ O(10³-10⁴) 操作/样本
# 生成1000个样本：~ 1-10秒（现代CPU）
```

### 空间复杂度
```python
# 内存使用
主要数组：
1. reflection_series: 1024 float32  ~ 4 KB
2. wavelet: ~500 float32            ~ 2 KB
3. synthetic_trace: 1024 float32    ~ 4 KB
4. segmentation_mask: 1024 int32    ~ 4 KB

# 总计：~ 14 KB/样本
# 批量生成1000样本：~ 14 MB
```


