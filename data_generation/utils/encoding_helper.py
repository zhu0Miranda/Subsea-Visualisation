import chardet
import os

def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding'], result['confidence']

def convert_config_encoding(config_path, target_encoding='utf-8'):
    """转换配置文件编码"""
    # 检测当前编码
    current_encoding, confidence = detect_encoding(config_path)
    print(f"检测到文件编码: {current_encoding} (置信度: {confidence:.2f})")
    
    if current_encoding.lower() == target_encoding.lower():
        print(f"文件已经是{target_encoding}编码，无需转换")
        return
    
    # 读取并转换
    try:
        with open(config_path, 'r', encoding=current_encoding) as f:
            content = f.read()
        
        # 备份原文件
        backup_path = config_path + '.backup'
        with open(backup_path, 'w', encoding=current_encoding) as f:
            f.write(content)
        print(f"已创建备份: {backup_path}")
        
        # 写入新编码
        with open(config_path, 'w', encoding=target_encoding) as f:
            f.write(content)
        print(f"已转换为{target_encoding}编码")
        
    except Exception as e:
        print(f"转换失败: {e}")

if __name__ == "__main__":
    config_file = "../config/simulation_params.yaml"
    if os.path.exists(config_file):
        convert_config_encoding(config_file)
    else:
        print(f"配置文件不存在: {config_file}")