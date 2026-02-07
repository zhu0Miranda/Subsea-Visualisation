#-*- coding: GB18030 -*-
import numpy as np
from scipy import signal
from typing import Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体（可选，根据系统调整）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 负号显示

class AcousticSimulator:
    """声学响应模拟器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sample_rate = config['simulation']['sample_rate']
        self.dt = 1.0 / self.sample_rate
    
    def generate_source_wavelet(self, pulse_type=None) -> np.ndarray:
        """生成声源子波"""
        if pulse_type is None:
            pulse_type = self.config['source']['pulse_type']
        freq = self.config['source']['frequency']
        duration = self.config['simulation']['duration']
        
        t = np.arange(0, duration, self.dt)
        
        if pulse_type == "ricker":
            # Ricker 子波 (Mexican hat wavelet)
            t0 = 1.0 / freq
            wavelet = (1 - 2 * (np.pi * freq * (t - t0)) ** 2) * \
                     np.exp(-(np.pi * freq * (t - t0)) ** 2)
        else:
            # 线性调频脉冲
            bandwidth = self.config['source'].get('bandwidth', freq * 0.5)
            t_frac = t / duration
            
            # 创建线性调频信号
            chirp_wave = signal.chirp(t, freq - bandwidth/2, duration, freq + bandwidth/2)
            
            # 使用汉宁窗口代替tukey窗口（兼容性更好）
            window = np.hanning(len(t))
            
            # 使用指数窗口使信号两端平滑
            if len(t) > 0:
                ramp_length = int(len(t) * 0.1)  # 10%的斜坡
                window[:ramp_length] = np.linspace(0, 1, ramp_length)
                window[-ramp_length:] = np.linspace(1, 0, ramp_length)
            
            wavelet = chirp_wave * window
        
        return wavelet / np.max(np.abs(wavelet)) if len(wavelet) > 0 else wavelet
    
    def convolutional_model(self, reflection_series: np.ndarray) -> np.ndarray:
        """一维卷积模型生成合成地震记录"""
        wavelet = self.generate_source_wavelet()
        
        # 进行卷积
        synthetic_trace = np.convolve(reflection_series, wavelet, mode='same')
        
        # 应用衰减
        synthetic_trace = self._apply_attenuation(synthetic_trace)
        
        return synthetic_trace
    
    def _apply_attenuation(self, trace: np.ndarray) -> np.ndarray:
        """应用地层衰减效应"""
        if len(trace) == 0:
            return trace
            
        # 简单的指数衰减模型
        t = np.arange(len(trace)) * self.dt
        attenuation_factor = np.exp(-0.5 * t)  # 简化衰减
        return trace * attenuation_factor
    
    def generate_3d_profile(self, reflection_series_2d: np.ndarray) -> np.ndarray:
        """生成3D剖面数据"""
        num_traces = self.config['simulation']['num_traces']
        trace_length = self.config['simulation']['trace_length']
        
        profile_3d = np.zeros((num_traces, num_traces, trace_length))
        wavelet = self.generate_source_wavelet()
        
        for i in range(num_traces):
            for j in range(num_traces):
                # 对每个道集应用卷积
                reflection_1d = reflection_series_2d[i, j, :] if reflection_series_2d.ndim == 3 else reflection_series_2d
                profile_3d[i, j, :] = np.convolve(reflection_1d, wavelet, mode='same')
        
        return profile_3d
    
    def visualize_wavelets(self):
        """可视化Ricker和Chirp子波"""
        # 生成两种子波
        ricker_wavelet = self.generate_source_wavelet(pulse_type="ricker")
        chirp_wavelet = self.generate_source_wavelet(pulse_type="chirp")
        
        # 时间轴
        duration = self.config['simulation']['duration']
        t = np.arange(0, duration, self.dt)
        
        # 创建图形
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 绘制Ricker子波
        axes[0].plot(t, ricker_wavelet, 'b-', linewidth=2)
        axes[0].set_title('Ricker子波波形', fontsize=14)
        axes[0].set_xlabel('时间 (s)', fontsize=12)
        axes[0].set_ylabel('振幅', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # 绘制Chirp子波
        axes[1].plot(t, chirp_wavelet, 'r-', linewidth=2)
        axes[1].set_title('Chirp子波波形', fontsize=14)
        axes[1].set_xlabel('时间 (s)', fontsize=12)
        axes[1].set_ylabel('振幅', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_reflection_series(self, reflection_series: np.ndarray):
        """可视化反射系数序列"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if reflection_series.ndim == 1:
            # 一维反射系数序列
            t = np.arange(len(reflection_series)) * self.dt
            axes[0].stem(t, reflection_series, linefmt='b-', markerfmt='bo', basefmt='r-')
            axes[0].set_title('一维反射系数序列', fontsize=14)
            axes[0].set_xlabel('时间 (s)', fontsize=12)
            axes[0].set_ylabel('反射系数', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            axes[1].set_visible(False)
            
        elif reflection_series.ndim == 2:
            # 二维反射系数剖面
            im = axes[0].imshow(reflection_series.T, aspect='auto', cmap='seismic', 
                               extent=[0, reflection_series.shape[0], 
                                       reflection_series.shape[1]*self.dt, 0])
            axes[0].set_title('二维反射系数剖面', fontsize=14)
            axes[0].set_xlabel('道号', fontsize=12)
            axes[0].set_ylabel('时间 (s)', fontsize=12)
            plt.colorbar(im, ax=axes[0], label='反射系数')
            
            # 显示部分道的波形
            num_traces_to_show = min(5, reflection_series.shape[0])
            colors = plt.cm.tab10(np.linspace(0, 1, num_traces_to_show))
            t = np.arange(reflection_series.shape[1]) * self.dt
            
            for i in range(num_traces_to_show):
                trace_idx = i * (reflection_series.shape[0] // num_traces_to_show)
                axes[1].plot(reflection_series[trace_idx, :], t, 
                           color=colors[i], linewidth=1.5, 
                           label=f'道号 {trace_idx}')
            
            axes[1].set_title('部分道反射系数', fontsize=14)
            axes[1].set_xlabel('反射系数', fontsize=12)
            axes[1].set_ylabel('时间 (s)', fontsize=12)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
            axes[1].invert_yaxis()  # 地震数据通常时间向下
            
        plt.tight_layout()
        plt.show()
    
    def visualize_2d_profile(self, profile_2d: np.ndarray):
        """可视化二维剖面"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 主剖面图
        vmax = np.max(np.abs(profile_2d)) * 0.5 if len(profile_2d) > 0 else 1
        im = axes[0, 0].imshow(profile_2d.T, aspect='auto', cmap='seismic', 
                              vmin=-vmax, vmax=vmax,
                              extent=[0, profile_2d.shape[0], 
                                      profile_2d.shape[1]*self.dt, 0])
        axes[0, 0].set_title('二维地震剖面', fontsize=14)
        axes[0, 0].set_xlabel('道号', fontsize=12)
        axes[0, 0].set_ylabel('时间 (s)', fontsize=12)
        plt.colorbar(im, ax=axes[0, 0], label='振幅')
        
        # 波形叠加显示
        axes[0, 1].plot(np.mean(profile_2d, axis=0), 
                       np.arange(profile_2d.shape[1]) * self.dt, 
                       'k-', linewidth=1)
        axes[0, 1].set_title('平均波形', fontsize=14)
        axes[0, 1].set_xlabel('振幅', fontsize=12)
        axes[0, 1].set_ylabel('时间 (s)', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].invert_yaxis()
        
        # 部分道的波形显示
        num_traces_to_show = min(5, profile_2d.shape[0])
        colors = plt.cm.tab10(np.linspace(0, 1, num_traces_to_show))
        t = np.arange(profile_2d.shape[1]) * self.dt
        
        for i in range(num_traces_to_show):
            trace_idx = i * (profile_2d.shape[0] // num_traces_to_show)
            axes[1, 0].plot(profile_2d[trace_idx, :], t, 
                          color=colors[i], linewidth=1.5, 
                          label=f'道号 {trace_idx}')
        
        axes[1, 0].set_title('部分道波形', fontsize=14)
        axes[1, 0].set_xlabel('振幅', fontsize=12)
        axes[1, 0].set_ylabel('时间 (s)', fontsize=12)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].invert_yaxis()
        
        # 振幅分布直方图
        axes[1, 1].hist(profile_2d.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 1].set_title('振幅分布直方图', fontsize=14)
        axes[1, 1].set_xlabel('振幅', fontsize=12)
        axes[1, 1].set_ylabel('频数', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_3d_profile(self, profile_3d: np.ndarray, slice_idx: int = None):
        """可视化三维剖面"""
        if slice_idx is None:
            slice_idx = profile_3d.shape[0] // 2
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. XY平面切片（时间切片）
        ax1 = fig.add_subplot(231)
        time_slice = profile_3d[:, :, slice_idx]
        im1 = ax1.imshow(time_slice, cmap='seismic', 
                        extent=[0, profile_3d.shape[1], 0, profile_3d.shape[0]])
        ax1.set_title(f'XY平面切片 (时间点 {slice_idx})', fontsize=12)
        ax1.set_xlabel('Y方向', fontsize=10)
        ax1.set_ylabel('X方向', fontsize=10)
        plt.colorbar(im1, ax=ax1, label='振幅')
        
        # 2. XZ平面切片
        ax2 = fig.add_subplot(232)
        xz_slice = profile_3d[slice_idx, :, :]
        im2 = ax2.imshow(xz_slice.T, aspect='auto', cmap='seismic',
                        extent=[0, profile_3d.shape[1], 
                                profile_3d.shape[2]*self.dt, 0])
        ax2.set_title(f'XZ平面切片 (X={slice_idx})', fontsize=12)
        ax2.set_xlabel('Y方向', fontsize=10)
        ax2.set_ylabel('时间 (s)', fontsize=10)
        plt.colorbar(im2, ax=ax2, label='振幅')
        
        # 3. YZ平面切片
        ax3 = fig.add_subplot(233)
        yz_slice = profile_3d[:, slice_idx, :]
        im3 = ax3.imshow(yz_slice.T, aspect='auto', cmap='seismic',
                        extent=[0, profile_3d.shape[0], 
                                profile_3d.shape[2]*self.dt, 0])
        ax3.set_title(f'YZ平面切片 (Y={slice_idx})', fontsize=12)
        ax3.set_xlabel('X方向', fontsize=10)
        ax3.set_ylabel('时间 (s)', fontsize=10)
        plt.colorbar(im3, ax=ax3, label='振幅')
        
        # 4. 三维可视化
        ax4 = fig.add_subplot(234, projection='3d')
        
        # 显示部分数据点以降低密度
        step = max(1, profile_3d.shape[0] // 20)
        X, Y = np.meshgrid(np.arange(0, profile_3d.shape[0], step),
                          np.arange(0, profile_3d.shape[1], step))
        
        # 提取振幅数据
        Z = profile_3d[::step, ::step, slice_idx]
        
        # 创建3D曲面
        surf = ax4.plot_surface(X, Y, Z, cmap='seismic', 
                               alpha=0.8, linewidth=0, antialiased=True)
        ax4.set_title('3D振幅曲面', fontsize=12)
        ax4.set_xlabel('X方向', fontsize=10)
        ax4.set_ylabel('Y方向', fontsize=10)
        ax4.set_zlabel('振幅', fontsize=10)
        
        # 5. 特定位置的道波形
        ax5 = fig.add_subplot(235)
        center_trace = profile_3d[slice_idx, slice_idx, :]
        t = np.arange(len(center_trace)) * self.dt
        ax5.plot(center_trace, t, 'b-', linewidth=2)
        ax5.set_title(f'中心道波形 (X={slice_idx}, Y={slice_idx})', 
                     fontsize=12)
        ax5.set_xlabel('振幅', fontsize=10)
        ax5.set_ylabel('时间 (s)', fontsize=10)
        ax5.grid(True, alpha=0.3)
        ax5.invert_yaxis()
        
        # 6. 振幅统计
        ax6 = fig.add_subplot(236)
        amplitudes = profile_3d.flatten()
        ax6.hist(amplitudes, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax6.set_title('三维数据振幅分布', fontsize=12)
        ax6.set_xlabel('振幅', fontsize=10)
        ax6.set_ylabel('频数', fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# 测试代码
if __name__ == "__main__":
    # 示例配置
    config = {
        'source': {
            'pulse_type': 'ricker',  # 或 'chirp'
            'frequency': 30,  # Hz
            'bandwidth': 20   # Hz (用于chirp)
        },
        'simulation': {
            'sample_rate': 1000,  # Hz
            'duration': 0.5,      # s
            'num_traces': 50,
            'trace_length': 500
        }
    }
    
    try:
        # 创建模拟器
        simulator = AcousticSimulator(config)
        
        print("测试子波生成...")
        # 1. 可视化子波
        simulator.visualize_wavelets()
        
        print("测试一维反射系数...")
        # 2. 生成并可视化反射系数
        # 一维示例
        reflection_1d = np.random.randn(500) * 0.5
        simulator.visualize_reflection_series(reflection_1d)
        
        print("测试二维反射系数...")
        # 二维示例
        reflection_2d = np.random.randn(50, 500) * 0.3
        simulator.visualize_reflection_series(reflection_2d)
        
        print("测试二维剖面...")
        # 3. 生成并可视化二维剖面
        # 注意：convolutional_model期望一维输入，这里需要调整
        # 生成模拟的二维剖面数据
        profile_2d = np.zeros((50, 500))
        wavelet = simulator.generate_source_wavelet()
        for i in range(50):
            reflection = np.random.randn(500) * 0.3
            profile_2d[i, :] = np.convolve(reflection, wavelet, mode='same')
        
        simulator.visualize_2d_profile(profile_2d)
        
        print("测试三维剖面...")
        # 4. 生成并可视化三维剖面
        reflection_3d = np.random.randn(50, 50, 500) * 0.2
        profile_3d = simulator.generate_3d_profile(reflection_3d)
        simulator.visualize_3d_profile(profile_3d)
        
        print("所有测试完成！")
        
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()