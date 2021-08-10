# 作者：Gen.D_
# 开发时间：2021/8/6 17:03
# 读取图片数据 格式为28*28
import scipy.misc as ms

# 文件路径
image_file_name = ""
img_array = ms.imread(image_file_name, flatten=True)
'''下一行代码重塑数组，将其从28×28的方块数组变成很长的一串数值，这是我们需要馈送给神经网络的数据。
此前，我们已经多次进行这样 的操作了。但是，这里比较新鲜的一点是将数组的值减去了255.0。
这样做 的原因是，常规而言，0指的是黑色，255指的是白色，但是，MNIST数据 集使用相反的方式表示，因此不得不将值逆转过来以匹配MNIST数据。'''
img_data = 255.0 - img_array.reshape(784)
img_data = (img_data / 255.0 * 0.99) + 0.01
