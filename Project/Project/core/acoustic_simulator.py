#-*- coding: GB18030 -*-
import numpy as np
from scipy import signal
from typing import Dict

class AcousticSimulator:
    """声学响应模拟器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sample_rate = config['simulation']['sample_rate']
        self.dt = 1.0 / self.sample_rate
    
    def generate_source_wavelet(self) -> np.ndarray:
        """生成声源子波"""
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
            bandwidth = self.config['source']['bandwidth']
            t_frac = t / duration
            chirp = signal.chirp(t, freq - bandwidth/2, duration, freq + bandwidth/2)
            window = signal.tukey(len(t), 0.2)
            wavelet = chirp * window
        
        return wavelet / np.max(np.abs(wavelet))
    
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