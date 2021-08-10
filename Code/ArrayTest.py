# 作者：Gen.D_
# 开发时间：2021/8/3 17:59
import numpy as np
import matplotlib.pyplot as pyplot
import pylab
# 创建一个3*2的数组
a = np.zeros([3,2])
# 设置数组值 默认以0为起始
a[0,0] = 1
a[1,1] = 1
a[2,0] = 1
# 绘制数组
pyplot.imshow(a,interpolation="nearest")
pylab.show()
# 随机生成numpy数组：结构3*3，数值为0~1的随机值
b = np.random.rand(3,3)

# 输出
print(a)
print(b)