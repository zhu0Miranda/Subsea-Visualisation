#-*- coding: GB18030 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

print( matplotlib.matplotlib_fname())

# 创建测试图形
plt.figure(figsize=(10, 6))

x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, 'r-', label='正弦曲线', linewidth=2)
plt.plot(x, y2, 'b--', label='余弦曲线', linewidth=2)

# 添加中文标签
plt.title('三角函数曲线图 - 测试中文字体', fontsize=16, fontweight='bold')
plt.xlabel('角度（弧度）', fontsize=14)
plt.ylabel('函数值', fontsize=14)
plt.legend(loc='best', fontsize=12)

# 添加网格和特殊字符测试
plt.grid(True, alpha=0.3)
plt.text(1.5, 0.8, '测试文字：αβγδε 中文 -123.45', 
         fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# 保存图片（验证字体是否嵌入）
plt.savefig('test_chinese.png', dpi=150, bbox_inches='tight')
print("图片已保存为 'test_chinese.png'，请检查中文是否正常显示")

plt.show()

# 打印当前配置以验证
print("\n当前字体配置：")
print(f"font.family: {plt.rcParams['font.family']}")
print(f"font.sans-serif: {plt.rcParams['font.sans-serif']}")
print(f"axes.unicode_minus: {plt.rcParams['axes.unicode_minus']}")
