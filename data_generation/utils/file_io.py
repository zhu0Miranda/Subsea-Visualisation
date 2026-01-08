#-*- coding: GB18030 -*-
import h5py
import numpy as np
import json
from datetime import datetime
from typing import Dict

class DataSaver:
    """数据保存类"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
    
    def save_training_sample(self, data: np.ndarray, labels: np.ndarray, 
                           metadata: Dict, filename: str):
        """保存训练样本到HDF5文件"""
        with h5py.File(f"{self.output_dir}/{filename}.h5", 'w') as f:
            # 保存数据
            f.create_dataset('sonar_data', data=data, compression='gzip')
            f.create_dataset('labels', data=labels, compression='gzip')
            
            # 保存元数据
            metadata_group = f.create_group('metadata')
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata_group.attrs[key] = value
                else:
                    metadata_group.attrs[key] = str(value)
            
            # 保存时间戳
            metadata_group.attrs['creation_time'] = datetime.now().isoformat()
    
    def save_dataset_info(self, dataset_info: Dict):
        """保存数据集信息"""
        with open(f"{self.output_dir}/dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)