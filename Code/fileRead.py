# 作者：Gen.D_
# 开发时间：2021/8/4 21:28
import numpy as np
import matplotlib.pyplot
import scipy.ndimage
import pylab

# 读文件
# data_file = open("D:\GoogleDownload\mnist_train_100.csv","r")
data_file = open("E:\我的收藏\图像识别\Python神经网络编程高清版\数据集\mnist_test_10.csv","r")
data_list = data_file.readlines()
data_file.close()

# 将文本转化为数组
# 将字符串分割后转化为字符串
all_values = data_list[0].split(',')
# print(type(all_values[0]))
# np.asfarray()：将文本字符串转化为float类型数字数组，并创建多维数组 |
image_array = np.asfarray(all_values[1:]).reshape((28,28))
# 矩阵旋转 逆时针
image_plus10_img = scipy.ndimage.rotate(image_array,10,cval=0.01,reshape=False)
# print(type(image_array))
# 绘图 | cmap=“Greys（灰度）
matplotlib.pyplot.imshow(image_plus10_img,cmap='Greys',interpolation=None)
pylab.show()