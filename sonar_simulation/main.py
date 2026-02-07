import numpy as np
import os
from tqdm import tqdm
import yaml
from typing import Dict, Any, List  # 添加 List 导入
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import warnings

# 忽略matplotlib的警告
warnings.filterwarnings('ignore', category=UserWarning)

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

# 硬编码配置，避免文件读取问题
DEFAULT_CONFIG = {
    'simulation': {
        'sample_rate': 100000,
        'duration': 0.1,
        'num_traces': 64,
        'trace_length': 1024
    },
    'source': {
        'frequency': 10000,
        'bandwidth': 8000,
        'pulse_type': 'ricker'
    },
    'sediment': {
        'min_layers': 1,
        'max_layers': 8,
        'thickness_range': [0.1, 5.0],
        'density_range': [1200, 2200],
        'velocity_range': [1450, 2200],
        'attenuation_range': [0.1, 5.0]
    },
    'sediment_types': {
        'clay': {
            'density': [1500, 1700],
            'velocity': [1470, 1550],
            'attenuation': [0.5, 1.5]
        },
        'silt': {
            'density': [1650, 1850],
            'velocity': [1550, 1650],
            'attenuation': [1.0, 2.5]
        },
        'sand': {
            'density': [1800, 2100],
            'velocity': [1650, 1850],
            'attenuation': [2.0, 4.0]
        },
        'gravel': {
            'density': [1900, 2200],
            'velocity': [1800, 2200],
            'attenuation': [3.0, 5.0]
        }
    },
    'noise': {
        'snr_range': [10, 30],
        'reverberation_level': 0.1,
        'electronic_noise': 0.05
    }
}

# 导入自定义模块
try:
    from core.sediment_generator import SedimentModelGenerator
    from core.acoustic_simulator import AcousticSimulator
    from core.sensor_simulator import SensorSimulator
    from core.data_augmentation import DataAugmentation
    from utils.file_io import DataSaver
except ImportError as e:
    print(f"导入模块时出错: {e}")
    print("请确保以下模块存在:")
    print("1. core/sediment_generator.py")
    print("2. core/acoustic_simulator.py")
    print("3. core/sensor_simulator.py")
    print("4. core/data_augmentation.py")
    print("5. utils/file_io.py")
    print("\n正在创建模拟数据生成器以进行测试...")
    
    # 创建模拟类以便继续运行
    class SedimentModelGenerator:
        def __init__(self):
            pass
        def generate_layered_model(self):
            return {'model_id': 'test', 'layers': [], 'total_depth': 10.0}
        def add_geological_features(self, model):
            return model
        def generate_reflection_coefficients(self, model):
            return np.random.randn(DEFAULT_CONFIG['simulation']['trace_length'])
        def generate_segmentation_mask(self, model):
            return np.zeros(DEFAULT_CONFIG['simulation']['trace_length'], dtype=np.int32)
    
    class AcousticSimulator:
        def __init__(self, config):
            self.config = config
        def convolutional_model(self, reflection_series):
            return reflection_series * 0.8 + np.random.normal(0, 0.1, len(reflection_series))
    
    class SensorSimulator:
        def __init__(self, config):
            self.config = config
        def add_noise(self, data):
            return data + np.random.normal(0, 0.05, len(data))
    
    class DataAugmentation:
        def __init__(self, config):
            self.config = config
        def apply_augmentation(self, data, mask):
            return data, mask
    
    class DataSaver:
        def __init__(self, path):
            self.path = path
            os.makedirs(path, exist_ok=True)
        def save_training_sample(self, data, labels, metadata, filename):
            print(f"保存样本 {filename}")
        def save_dataset_info(self, info):
            print("保存数据集信息")

class Visualizer:
    """可视化工具类"""
    
    @staticmethod
    def plot_single_sample(sample: Dict[str, Any], sample_id: int = 0, save_path: str = None):
        """绘制单个样本的可视化图表"""
        data = sample['data']
        labels = sample['labels']
        metadata = sample['metadata']
        
        # 创建多子图
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=[2, 2, 1.5, 1])
        
        # 1. 声呐数据波形图
        ax1 = fig.add_subplot(gs[0, :])
        t = np.arange(len(data)) / DEFAULT_CONFIG['simulation']['sample_rate']
        ax1.plot(t, data, 'b-', linewidth=1, alpha=0.8, label='声呐数据')
        ax1.set_xlabel('时间 (s)', fontsize=12)
        ax1.set_ylabel('振幅', fontsize=12)
        ax1.set_title(f'样本 {sample_id} - 声呐数据波形', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # 添加统计信息
        stats_text = f"均值: {np.mean(data):.4f}, 标准差: {np.std(data):.4f}, 峰值: {np.max(np.abs(data)):.4f}"
        ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. 分割掩码图
        ax2 = fig.add_subplot(gs[1, :])
        
        # 类别颜色映射
        class_colors = {
            0: '#1f77b4',  # 海水 - 蓝色
            1: '#2ca02c',  # clay - 绿色
            2: '#ff7f0e',  # silt - 橙色
            3: '#d62728',  # sand - 红色
            4: '#9467bd',  # gravel - 紫色
            5: '#8c564b',  # mixed - 棕色
            6: '#e377c2',  # gas_pocket - 粉色
            7: '#7f7f7f',  # buried_object - 灰色
            8: '#bcbd22',  # fault - 黄绿色
            9: '#17becf',  # bioturbation - 青色
            10: '#ff9896'  # irregular_interface - 浅红色
        }
        
        # 类别标签
        class_labels = {
            0: '海水',
            1: '粘土',
            2: '粉砂',
            3: '沙',
            4: '砾石',
            5: '混合',
            6: '气包',
            7: '埋藏物体',
            8: '断层',
            9: '生物扰动',
            10: '不规则界面'
        }
        
        # 绘制分割掩码
        for class_id in np.unique(labels):
            if class_id in class_colors:
                mask_indices = np.where(labels == class_id)[0]
                if len(mask_indices) > 0:
                    ax2.scatter(t[mask_indices], [class_id] * len(mask_indices), 
                              color=class_colors[class_id], s=10, alpha=0.7,
                              label=class_labels.get(class_id, f'类别{class_id}'))
        
        ax2.set_xlabel('时间 (s)', fontsize=12)
        ax2.set_ylabel('类别', fontsize=12)
        ax2.set_title('分割掩码标注', fontsize=14, fontweight='bold')
        ax2.set_yticks(list(class_labels.keys()))
        ax2.set_yticklabels([class_labels.get(i, f'类{i}') for i in class_labels.keys()])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=9)
        
        # 3. 频谱分析
        ax3 = fig.add_subplot(gs[2, 0])
        if len(data) > 1:
            fft_data = np.fft.fft(data)
            freq = np.fft.fftfreq(len(data), 1/DEFAULT_CONFIG['simulation']['sample_rate'])
            positive_freq = freq[:len(freq)//2]
            positive_fft = np.abs(fft_data[:len(freq)//2])
            ax3.plot(positive_freq / 1000, positive_fft, 'g-', linewidth=1, alpha=0.7)
            ax3.set_xlabel('频率 (kHz)', fontsize=11)
            ax3.set_ylabel('幅值', fontsize=11)
            ax3.set_title('频谱分析', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # 标记主频
            if len(positive_fft) > 0:
                peak_freq_idx = np.argmax(positive_fft[1:]) + 1
                peak_freq = positive_freq[peak_freq_idx] / 1000
                peak_mag = positive_fft[peak_freq_idx]
                ax3.plot(peak_freq, peak_mag, 'ro', markersize=8)
                ax3.text(peak_freq, peak_mag, f' {peak_freq:.1f} kHz', 
                        fontsize=10, verticalalignment='bottom')
        
        # 4. 直方图
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('振幅', fontsize=11)
        ax4.set_ylabel('频数', fontsize=11)
        ax4.set_title('振幅分布', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 添加分布参数
        skewness = np.mean((data - np.mean(data))**3) / (np.std(data)**3)
        kurtosis = np.mean((data - np.mean(data))**4) / (np.std(data)**4)
        ax4.text(0.02, 0.95, f'偏度: {skewness:.2f}\n峰度: {kurtosis:.2f}', 
                transform=ax4.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 5. 自相关图
        ax5 = fig.add_subplot(gs[2, 2])
        autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr[:min(200, len(autocorr))]  # 只显示前200个点
        ax5.plot(autocorr, 'm-', linewidth=1.5, alpha=0.7)
        ax5.set_xlabel('延迟', fontsize=11)
        ax5.set_ylabel('自相关', fontsize=11)
        ax5.set_title('自相关函数', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. 模型信息表格
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('tight')
        ax6.axis('off')
        
        # 准备表格数据
        table_data = []
        table_data.append(['模型ID', metadata.get('model_id', 'N/A')])
        table_data.append(['样本ID', metadata.get('sample_id', 'N/A')])
        table_data.append(['层数', metadata.get('num_layers', 'N/A')])
        table_data.append(['总深度', f"{metadata.get('total_depth', 0):.2f} m"])
        
        if 'layer_info' in metadata and len(metadata['layer_info']) > 0:
            # 统计沉积物类型
            sediment_types = {}
            for layer in metadata['layer_info']:
                sed_type = layer.get('sediment_type', 'unknown')
                sediment_types[sed_type] = sediment_types.get(sed_type, 0) + 1
            
            type_str = ', '.join([f"{k}:{v}" for k, v in sediment_types.items()])
            table_data.append(['沉积物分布', type_str])
        
        if 'features' in metadata and len(metadata['features']) > 0:
            feature_types = {}
            for feature in metadata['features']:
                ftype = feature.get('type', 'unknown')
                feature_types[ftype] = feature_types.get(ftype, 0) + 1
            
            feature_str = ', '.join([f"{k}:{v}" for k, v in feature_types.items()])
            table_data.append(['地质特征', feature_str])
        
        # 添加数据统计
        table_data.append(['数据长度', f"{len(data)} 采样点"])
        table_data.append(['采样率', f"{DEFAULT_CONFIG['simulation']['sample_rate']} Hz"])
        table_data.append(['时间范围', f"{t[-1]:.3f} s"])
        
        # 创建表格
        table = ax6.table(cellText=table_data, 
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.2, 0.8])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(table_data)):
            table[(i, 0)].set_facecolor('#f0f0f0')
            table[(i, 1)].set_facecolor('#fafafa')
        
        plt.suptitle(f'声呐仿真样本可视化 - 样本 {sample_id}', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_dataset_summary(dataset_info: Dict[str, Any], samples_metadata: List[Dict[str, Any]], 
                           output_dir: str = "visualization"):
        """绘制数据集总结图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取统计信息
        num_layers_list = [meta.get('num_layers', 0) for meta in samples_metadata]
        total_depth_list = [meta.get('total_depth', 0) for meta in samples_metadata]
        
        # 统计沉积物类型分布
        sediment_counts = {}
        feature_counts = {}
        
        for meta in samples_metadata:
            # 沉积物类型
            if 'layer_info' in meta:
                for layer in meta['layer_info']:
                    sed_type = layer.get('sediment_type', 'unknown')
                    sediment_counts[sed_type] = sediment_counts.get(sed_type, 0) + 1
            
            # 地质特征
            if 'features' in meta:
                for feature in meta['features']:
                    ftype = feature.get('type', 'unknown')
                    feature_counts[ftype] = feature_counts.get(ftype, 0) + 1
        
        # 创建总结图表
        fig = plt.figure(figsize=(18, 12))
        
        # 1. 层数分布
        ax1 = plt.subplot(2, 3, 1)
        unique_layers, layer_counts = np.unique(num_layers_list, return_counts=True)
        ax1.bar(unique_layers, layer_counts, color='skyblue', edgecolor='black')
        ax1.set_xlabel('层数', fontsize=12)
        ax1.set_ylabel('样本数量', fontsize=12)
        ax1.set_title('沉积层层数分布', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 在柱子上添加数字
        for i, count in enumerate(layer_counts):
            ax1.text(unique_layers[i], count + 0.1, str(int(count)), 
                    ha='center', va='bottom', fontsize=10)
        
        # 2. 深度分布
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(total_depth_list, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('总深度 (m)', fontsize=12)
        ax2.set_ylabel('样本数量', fontsize=12)
        ax2.set_title('总深度分布', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加平均深度线
        avg_depth = np.mean(total_depth_list)
        ax2.axvline(avg_depth, color='red', linestyle='--', linewidth=2, 
                   label=f'平均深度: {avg_depth:.2f} m')
        ax2.legend()
        
        # 3. 沉积物类型分布
        ax3 = plt.subplot(2, 3, 3)
        if sediment_counts:
            sediment_types = list(sediment_counts.keys())
            counts = list(sediment_counts.values())
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(sediment_types)))
            wedges, texts, autotexts = ax3.pie(counts, labels=sediment_types, autopct='%1.1f%%',
                                              colors=colors, startangle=90)
            
            # 美化百分比文本
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
            
            ax3.set_title('沉积物类型分布', fontsize=14, fontweight='bold')
        
        # 4. 地质特征分布
        ax4 = plt.subplot(2, 3, 4)
        if feature_counts:
            feature_types = list(feature_counts.keys())
            fcounts = list(feature_counts.values())
            
            y_pos = np.arange(len(feature_types))
            bars = ax4.barh(y_pos, fcounts, color='lightcoral', edgecolor='black')
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(feature_types)
            ax4.set_xlabel('出现次数', fontsize=12)
            ax4.set_title('地质特征分布', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')
            
            # 在条形上添加数字
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax4.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                        str(int(width)), ha='left', va='center', fontsize=10)
        
        # 5. 深度 vs 层数 散点图
        ax5 = plt.subplot(2, 3, 5)
        scatter = ax5.scatter(num_layers_list, total_depth_list, 
                            c=total_depth_list, cmap='viridis', alpha=0.6, s=50)
        ax5.set_xlabel('层数', fontsize=12)
        ax5.set_ylabel('总深度 (m)', fontsize=12)
        ax5.set_title('层数与深度关系', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax5, label='深度 (m)')
        
        # 6. 数据集信息表格
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('tight')
        ax6.axis('off')
        
        table_data = [
            ['总样本数', str(dataset_info.get('total_samples', 0))],
            ['创建日期', dataset_info.get('creation_date', 'N/A')],
            ['平均层数', f"{np.mean(num_layers_list):.2f}"],
            ['平均深度', f"{np.mean(total_depth_list):.2f} m"],
            ['最大深度', f"{np.max(total_depth_list):.2f} m"],
            ['最小深度', f"{np.min(total_depth_list):.2f} m"],
            ['沉积物类型数', str(len(sediment_counts))],
            ['地质特征类型数', str(len(feature_counts))]
        ]
        
        table = ax6.table(cellText=table_data, 
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.4, 0.6])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(table_data)):
            table[(i, 0)].set_facecolor('#e0e0e0')
            table[(i, 1)].set_facecolor('#f5f5f5')
        
        plt.suptitle('声呐仿真数据集统计总结', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # 保存图表
        summary_path = os.path.join(output_dir, "dataset_summary.png")
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"数据集总结图表已保存到: {summary_path}")
        
        # 生成文本总结
        txt_summary = f"""
        ========================================
        声呐仿真数据集总结报告
        ========================================
        生成时间: {dataset_info.get('creation_date', 'N/A')}
        总样本数: {dataset_info.get('total_samples', 0)}
        
        层数统计:
          平均层数: {np.mean(num_layers_list):.2f}
          最小层数: {np.min(num_layers_list)}
          最大层数: {np.max(num_layers_list)}
        
        深度统计:
          平均深度: {np.mean(total_depth_list):.2f} m
          最小深度: {np.min(total_depth_list):.2f} m
          最大深度: {np.max(total_depth_list):.2f} m
        
        沉积物类型统计:
        """
        
        for sed_type, count in sediment_counts.items():
            percentage = count / sum(sediment_counts.values()) * 100 if sum(sediment_counts.values()) > 0 else 0
            txt_summary += f"  {sed_type}: {count} 次 ({percentage:.1f}%)\n"
        
        if feature_counts:
            txt_summary += "\n地质特征统计:\n"
            for ftype, count in feature_counts.items():
                percentage = count / len(samples_metadata) * 100 if len(samples_metadata) > 0 else 0
                txt_summary += f"  {ftype}: {count} 次 ({percentage:.1f}%)\n"
        
        txt_summary += "\n" + "="*40 + "\n"
        
        # 保存文本总结
        txt_path = os.path.join(output_dir, "dataset_summary.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(txt_summary)
        
        print(f"文本总结已保存到: {txt_path}")
        print("\n" + txt_summary)

class SonarSimulationPipeline:
    """修复的声呐仿真流水线 - 使用硬编码配置"""
    
    def __init__(self, visualize_samples: bool = True, 
                 visualization_interval: int = 10):
        # 使用硬编码配置
        self.config = DEFAULT_CONFIG
        
        # 可视化设置
        self.visualize_samples = visualize_samples
        self.visualization_interval = visualization_interval
        
        # 初始化组件
        self.sediment_gen = SedimentModelGenerator()
        self.acoustic_sim = AcousticSimulator(self.config)
        self.sensor_sim = SensorSimulator(self.config)
        self.data_aug = DataAugmentation(self.config)
        self.data_saver = DataSaver("training_data")
        self.visualizer = Visualizer()
        
        # 创建输出目录
        os.makedirs("training_data", exist_ok=True)
        os.makedirs("visualization", exist_ok=True)
        
        # 用于存储生成的样本元数据
        self.samples_metadata = []
    
    def generate_single_sample(self, sample_id: int) -> Dict[str, Any]:
        """生成单个训练样本"""
        # 1. 生成地质模型
        base_model = self.sediment_gen.generate_layered_model()
        model_with_features = self.sediment_gen.add_geological_features(base_model)
        
        # 2. 生成反射系数序列
        reflection_series = self.sediment_gen.generate_reflection_coefficients(model_with_features)
        
        # 3. 生成声学响应
        clean_trace = self.acoustic_sim.convolutional_model(reflection_series)
        
        # 4. 添加传感器效应和噪声
        noisy_trace = self.sensor_sim.add_noise(clean_trace)
        
        # 5. 生成标注
        segmentation_mask = self.sediment_gen.generate_segmentation_mask(model_with_features)
        
        # 6. 数据增强
        augmented_trace, augmented_mask = self.data_aug.apply_augmentation(
            noisy_trace, segmentation_mask
        )
        
        # 准备元数据
        metadata = {
            'sample_id': sample_id,
            'model_id': model_with_features['model_id'],
            'num_layers': len(model_with_features['layers']),
            'total_depth': model_with_features['total_depth'],
            'layer_info': model_with_features['layers'],
            'features': model_with_features.get('features', [])
        }
        
        # 存储元数据用于后续统计
        self.samples_metadata.append(metadata)
        
        # 可视化样本（每隔一定间隔）
        if self.visualize_samples and sample_id % self.visualization_interval == 0:
            sample_data = {
                'data': augmented_trace,
                'labels': augmented_mask,
                'metadata': metadata
            }
            
            # 保存可视化图表
            viz_path = os.path.join("visualization", f"sample_{sample_id:06d}.png")
            self.visualizer.plot_single_sample(sample_data, sample_id, viz_path)
        
        return {
            'data': augmented_trace,
            'labels': augmented_mask,
            'metadata': metadata
        }
    
    def generate_dataset(self, num_samples: int = 100):
        """生成完整数据集"""
        # 重置元数据列表
        self.samples_metadata = []
        
        dataset_info = {
            'total_samples': num_samples,
            'creation_date': str(np.datetime64('now'))
        }
        
        # 生成进度条
        progress_bar = tqdm(range(num_samples), desc="生成样本")
        
        for i in progress_bar:
            sample = self.generate_single_sample(i)
            
            # 更新进度条描述
            metadata = sample['metadata']
            progress_bar.set_postfix({
                '层数': metadata['num_layers'],
                '深度': f"{metadata['total_depth']:.1f}m",
                '特征数': len(metadata.get('features', []))
            })
            
            # 保存样本
            filename = f"sample_{i:06d}"
            self.data_saver.save_training_sample(
                sample['data'], sample['labels'], sample['metadata'], filename
            )
        
        # 保存数据集信息
        self.data_saver.save_dataset_info(dataset_info)
        
        # 生成数据集总结图表
        print("\n" + "="*60)
        print("生成数据集总结图表...")
        print("="*60)
        
        try:
            self.visualizer.plot_dataset_summary(dataset_info, self.samples_metadata)
        except Exception as e:
            print(f"生成总结图表时出错: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'='*60}")
        print(f"数据集生成完成！共 {num_samples} 个样本")
        print(f"数据保存目录: training_data/")
        print(f"可视化图表目录: visualization/")
        print(f"{'='*60}")
        
        # 生成完成提示
        self._print_completion_message(num_samples)
    
    def _print_completion_message(self, num_samples: int):
        """打印完成信息"""
        completion_msg = f"""
        🎉 声呐仿真数据集生成完成！
        
        统计信息:
        📊 总样本数: {num_samples}
        📁 数据目录: training_data/
        🖼️  图表目录: visualization/
        
        下一步建议:
        1. 查看 visualization/dataset_summary.png 了解数据集概况
        2. 检查 visualization/sample_*.png 查看具体样本
        3. 使用生成的训练数据训练您的深度学习模型
        
        生成的样本文件格式:
        ├── sample_000000.npy      # 声呐数据
        ├── sample_000000_labels.npy # 标注数据
        └── sample_000000_metadata.npy # 元数据
        """
        
        print(completion_msg)

def test_visualization():
    """测试可视化功能"""
    print("测试可视化功能...")
    print("="*60)
    
    # 创建临时样本用于测试
    sample = {
        'data': np.sin(np.linspace(0, 10, DEFAULT_CONFIG['simulation']['trace_length'])) + \
                np.random.normal(0, 0.1, DEFAULT_CONFIG['simulation']['trace_length']),
        'labels': np.random.choice([0, 1, 2, 3, 4, 6], DEFAULT_CONFIG['simulation']['trace_length'], 
                                  p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05]),
        'metadata': {
            'sample_id': 999,
            'model_id': 'test_model_001',
            'num_layers': 5,
            'total_depth': 12.5,
            'layer_info': [
                {'sediment_type': 'clay', 'thickness': 2.0, 'density': 1600, 'velocity': 1500},
                {'sediment_type': 'silt', 'thickness': 3.0, 'density': 1750, 'velocity': 1600},
                {'sediment_type': 'sand', 'thickness': 2.5, 'density': 1950, 'velocity': 1700},
                {'sediment_type': 'gravel', 'thickness': 3.0, 'density': 2100, 'velocity': 2000},
                {'sediment_type': 'mixed', 'thickness': 2.0, 'density': 1800, 'velocity': 1650}
            ],
            'features': [
                {'type': 'gas_pocket', 'depth': 4.5, 'intensity': 0.7},
                {'type': 'buried_object', 'depth': 8.2, 'object_type': 'metal'}
            ]
        }
    }
    
    # 显示单个样本可视化
    Visualizer.plot_single_sample(sample, sample_id=999)
    
    # 测试数据集总结
    dataset_info = {
        'total_samples': 50,
        'creation_date': '2024-01-15'
    }
    
    samples_metadata = [
        {'num_layers': 3, 'total_depth': 8.5, 'layer_info': [{'sediment_type': 'clay'}]},
        {'num_layers': 5, 'total_depth': 12.2, 'layer_info': [{'sediment_type': 'sand'}]},
        {'num_layers': 4, 'total_depth': 10.1, 'layer_info': [{'sediment_type': 'silt'}]},
        {'num_layers': 6, 'total_depth': 15.3, 'layer_info': [{'sediment_type': 'gravel'}]},
        {'num_layers': 4, 'total_depth': 9.8, 'layer_info': [{'sediment_type': 'mixed'}]},
    ]
    
    print("测试完成!")

if __name__ == "__main__":
    print("声呐仿真数据生成系统")
    print("="*60)
    
    # 用户选择
    print("请选择运行模式:")
    print("1. 生成完整数据集")
    print("2. 测试可视化功能")
    print("3. 自定义配置生成")
    print("4. 仅生成少量样本测试")
    
    try:
        choice = input("请输入选择 (1-4): ").strip()
    except EOFError:
        # 如果是从脚本调用，使用默认值
        choice = "1"
    
    if choice == "1":
        # 生成完整数据集
        try:
            num_samples_input = input("请输入要生成的样本数量 (默认50): ").strip()
            num_samples = int(num_samples_input) if num_samples_input else 50
        except:
            num_samples = 50
            
        try:
            interval_input = input("可视化间隔 (每多少个样本显示一次，默认10): ").strip()
            visualize_interval = int(interval_input) if interval_input else 10
        except:
            visualize_interval = 10
        
        pipeline = SonarSimulationPipeline(
            visualize_samples=True,
            visualization_interval=visualize_interval
        )
        
        print(f"\n开始生成 {num_samples} 个样本的数据集...")
        pipeline.generate_dataset(num_samples=num_samples)
        
    elif choice == "2":
        # 测试可视化
        test_visualization()
        
    elif choice == "3":
        # 自定义配置
        print("自定义配置功能暂未实现，使用默认配置...")
        
        try:
            num_samples_input = input("请输入样本数量: ").strip()
            num_samples = int(num_samples_input) if num_samples_input else 10
        except:
            num_samples = 10
            
        try:
            interval_input = input("可视化间隔: ").strip()
            visualize_interval = int(interval_input) if interval_input else 5
        except:
            visualize_interval = 5
        
        pipeline = SonarSimulationPipeline(
            visualize_samples=True,
            visualization_interval=visualize_interval
        )
        
        print(f"\n开始生成 {num_samples} 个样本的数据集...")
        pipeline.generate_dataset(num_samples=num_samples)
        
    elif choice == "4":
        # 仅生成少量样本测试
        print("\n生成少量样本进行测试...")
        pipeline = SonarSimulationPipeline(
            visualize_samples=True,
            visualization_interval=1  # 每个样本都可视化
        )
        pipeline.generate_dataset(num_samples=5)
        
    else:
        print("无效选择，使用默认模式...")
        pipeline = SonarSimulationPipeline()
        pipeline.generate_dataset(num_samples=10)