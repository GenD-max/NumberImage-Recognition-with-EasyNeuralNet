import numpy as np
import scipy.special, scipy.ndimage
import matplotlib.pyplot
import pylab


# 定义神经网络类
class neuralNetwork:

    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 设置输入层节点、隐藏层节点、输出层节点
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 设置链接权重
        # numpy.random.normal(loc=0.0, scale=1.0, size=None)
        # loc：分布的均值（中心）| scale：分布的标准差（宽度）| size：输出的维度
        # pow(x,y)：x**y
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 设置学习率
        self.lr = learningrate

        # lambda x: scipy.special.expit(x)：创建激活函数S函数
        self.activation_function = lambda x: scipy.special.expit(x)
        # 逆激活函数S
        self.inverse_activation_function = lambda x: scipy.special.logit(x)
        pass

    # 训练神经网络
    def train(self, inputs_list, targets_list):
        # 将输入列表转化为二维数组
        # np.array(inputs_list,ndmin = 2).T：  ndmin为维数   ”.T为“矩阵转置
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 计算进入隐藏层的信号
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算从隐藏层输出的信号
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算从最终层输出的信号
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算最终层输出的最终信号
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        # 按权重分割，得到隐藏层误差
        hidden_errors = np.dot(self.who.T, output_errors)

        # 更新隐藏层和输出层之间的链接权重：见书本P75公式
        # np.transpose()：矩阵转置，并且适用于多维数组，因为who在上面被转置了
        # .T：矩阵转置，适用于一、二维数组
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))

        # 更新输入层于隐藏层之间的链接权重
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass

    # 查询神经网络
    def query(self, inputs_list):
        # 将列表转化为二维数组
        inputs = np.array(inputs_list, ndmin=2).T
        # 计算信号进入隐藏层：Xhidden = Winput_hidden * I
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算从隐藏层输出的信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算信号进入最终输出层
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算从最终输出层出现的信号
        final_outputs = self.activation_function(final_inputs)

        # 返回最终信号
        return final_outputs

    # 反查询神经网络
    '''
    我们将对每个项目使用相同的术语，
    例如，目标是网络右侧的值，尽管用作输入
    例如 hidden_output 是中间节点右侧的信号
    '''

    def backquery(self, targets_list):
        # 将目标列表转置为垂直数组
        final_outputs = np.array(targets_list, ndmin=2).T

        # 计算进入最终输出层的信号
        final_inputs = self.inverse_activation_function(final_outputs)

        # 计算出隐藏层的信号
        # 原来who * hidden_outputs = final_inputs
        # 现在hidden_outputs = who.T * final_inputs
        hidden_outputs = np.dot(self.who.T, final_inputs)

        # 将它们缩放回 0.01 到 0.99
        '''
        逻辑S函数接受了任何数值，输出0和1之间的某个值，但是不包括0和1本身。
        逆函数必须接受相同的范围0和1之间的某个值，不包括0和1，弹出任何正值或负值。
        为了实现这一目标，我们简单地 接受输出层中的所有值，应用logit()，并将它们重新调整到有效范围。
        我选 择的范围为0.01至0.99。
        '''
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # 计算信号进入隐藏层
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # 计算输入层外的信号
        inputs = np.dot(self.wih.T, hidden_inputs)
        # 将它们缩放回 0.01 到 0.99
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        # 返回多维数组(单行)
        return inputs


# 各层节点数量
input_nodes = 784
hidden_nodes = 200  # 影响准确率
output_nodes = 10
# 学习率
learning_rate = 0.1  # 影响准确率
# 训练次数
epochs = 1  # 影响准确率
# 构建神经网络实例
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# 读取训练数据
# training_data_file = open("E:\我的收藏\图像识别\Python神经网络编程高清版\数据集\mnist_train_100.csv", "r")
training_data_file = open("E:\我的收藏\图像识别\Python神经网络编程高清版\数据集\mnist_train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()
# 训练神经网络
for e in range(epochs):
    for record in training_data_list:
        # 将字符串分割后转化为列表
        all_values = record.split(',')
        # np.asfarray()：将文本字符串转化为实数，并初始化限制范围
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # 创建目标输出值（全部为 0.01，除了所需的标签为 0.99）
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] 是这条记录的目标标签，也就是标签的值，该图像真实表达的值
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        # 创建旋转变体
        # 逆时针旋转 10 度
        inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), 10, cval=0.01, order=1,
                                                              reshape=False)
        # 训练
        n.train(inputs_plusx_img.reshape(784), targets)
        # 顺时针旋转 10 度
        inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), -10, cval=0.01, order=1,
                                                               reshape=False)
        # 训练
        n.train(inputs_minusx_img.reshape(784), targets)
        pass

# 读取测试数据
# test_data_file = open("E:\我的收藏\图像识别\Python神经网络编程高清版\数据集\mnist_test_10.csv", 'r')
test_data_file = open("E:\我的收藏\图像识别\Python神经网络编程高清版\数据集\mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
# 验证的记分卡，最初为空
scorecard = []
# 测试神经网络
for record in test_data_list:
    # 预处理
    test_value = record.split(',')
    # 得到标签数据
    corrent_label = int(test_value[0])
    # print("测试标签数字：", corrent_label)
    # 处理，缩小范围
    test_inputs = (np.asfarray(test_value[1:]) / 255.0 * 0.99) + 0.01
    # 查询网络
    test_outputs = n.query(test_inputs)
    # 最高值的索引对应于标签
    label = np.argmax(test_outputs)
    # print("查询的结果为：", label)
    # 计分板操作：正确得1分，错误得0分
    if (label == corrent_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass
# 计算准确率
scorecard_array = np.asarray(scorecard)  # 将列表转化为多维数组
# print(type(scorecard_array))
network_accuracyrate = scorecard_array.sum() / scorecard_array.size
print("得分板：", scorecard)
print("准确率：", network_accuracyrate)

# 向后运行网络，给定一个标签，看看它产生了什么图像
# 要测试的标签
label = 8
# 为这个标签创建输出信号
targets = np.zeros(output_nodes) + 0.01
# all_values[0] 是这条记录的目标标签
targets[label] = 0.99
print("向后运行网络的目标数组：", targets)
# 获取图像数据
image_data = n.backquery(targets)
# 绘制图像数据
matplotlib.pyplot.imshow(image_data.reshape(28, 28), cmap='Greys', interpolation='None')
pylab.show()
