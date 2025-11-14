import numpy  # 测试 numpy 导入
import matplotlib.pyplot as plt  # 测试 matplotlib 导入

# 测试 numpy + matplotlib 协同绘图
x = numpy.linspace(0, 10, 100)
y = numpy.sin(x)
plt.plot(x, y)
plt.title("Test Plot")
plt.show()