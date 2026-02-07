#-*- coding: GB18030 -*-
import os
import sys
import subprocess

def check_environment():
    """检查运行环境"""
    print("=== 声呐仿真项目环境检查 ===")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要包
    required_packages = ['numpy', 'scipy', 'matplotlib', 'h5py', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少以下包: {', '.join(missing_packages)}")
        install = input("是否尝试安装? (y/n): ")
        if install.lower() == 'y':
            for package in missing_packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # 检查配置文件
    config_path = "config/simulation_params.yaml"
    if os.path.exists(config_path):
        print(f"✓ 配置文件存在: {config_path}")
        
    else:
        print(f"✗ 配置文件不存在: {config_path}")
    


    print("\n" + "="*50)

def main():
    """主函数"""
    check_environment()
    
    print("\n请选择运行模式:")
    print("1. 完整仿真 (使用配置文件)")
    print("2. 简化演示 (不依赖配置文件)")
    print("3. 生成训练数据集")
    
    choice = input("\n请输入选择 (1-3): ").strip()
    
    if choice == "1":
        print("\n运行完整仿真...")
        try:
            from main import SonarSimulationPipeline
            pipeline = SonarSimulationPipeline()
            #pipeline.generate_dataset(num_samples=10, use_3d=False)
            pipeline.generate_dataset(num_samples=10)
        except Exception as e:
            print(f"完整仿真失败: {e}")
            #print("尝试运行简化演示...")
            #subprocess.run([sys.executable, "examples/simple_demo.py"])
    
    elif choice == "2":
        print("\n运行简化演示...")
        subprocess.run([sys.executable, "examples/simple_demo.py"])
    
    elif choice == "3":
        print("\n生成训练数据集...")
        try:
            from main_fixed import FixedSonarSimulationPipeline
            pipeline = FixedSonarSimulationPipeline()
            num_samples = int(input("请输入要生成的样本数量 (默认100): ") or "100")
            pipeline.generate_dataset(num_samples=num_samples)
        except Exception as e:
            print(f"生成数据集失败: {e}")
    
    
    else:
        print("无效选择")

if __name__ == "__main__":
    main()