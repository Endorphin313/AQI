import matplotlib
import pandas as pd
import numpy as np
import sns
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader,Dataset
import seaborn as sns
# 导入PyTorch中的Dataset类，用于自定义数据集
from torch.utils.data import Dataset

# 定义一个函数来获取数据，进行预处理，最后返回用于训练和测试的数据加载器
def getData(root, sequence_length, batch_size):
    # 使用pandas读取指定路径的CSV文件
    stock_data = pd.read_csv(root)
    # 打印数据框架的信息，包括每列的数据类型和非空值数量，帮助了解数据结构
    # print(stock_data.info())
    # 打印数据的前五行，以便初步查看数据内容
    # print(stock_data.head())
    # 删除数据中不需要的列
    #stock_data.drop('date', axis=1, inplace=True)
    # 再次打印处理后的数据的前五行，确认已正确删除不需要的列
    # print("整理后\n", stock_data.head())

    # 获取预测列的最大值和最小值，用于后续的数据标准化处理
    close_max = stock_data["AQI"].max()
    close_min = stock_data["AQI"].min()

    # 初始化MinMaxScaler，将数据标准化到0和1之间
    scaler = MinMaxScaler()
    # 对数据进行标准化处理
    df = scaler.fit_transform(stock_data)

    # 打印标准化后的数据形状，确认数据结构
    # print("整理后\n", stock_data.shape)

    # 根据指定的序列长度构造输入X和输出Y，用于训练模型
    sequence = sequence_length
    x = []  # 存储输入序列
    y = []  # 存储输出值（目标值）
    # 遍历数据，根据sequence_length构造输入和输出
    for i in range(df.shape[0] - sequence):
        x.append(df[i:i + sequence, :])   #
        y.append(df[i + sequence, 0])  # 预测下一小时的空气质量

    # 将输入和输出列表转换为numpy数组，方便后续处理
    x = np.array(x, dtype=np.float32)  #将x转换成一个NumPy数组，并且指定数组中元素的数据类型为float32。
    y = np.array(y, dtype=np.float32).reshape(-1, 1)  #使用.reshape(-1, 1)方法改变数组的形状。第一个维度-1表示该维度的大小将自动计算以便满足总元素数量不变的原则，
                    # 第二个维度1表示每个内部列表包含一个元素。简单来说，这会把y变成一个列向量，无论原来y的形状如何，确保它最终是一个二维数组，其中每行只有一个元素。

    # 打印输入和输出数据的形状，以便检查
    # print("x.shape=", x.shape)
    # print("y.shape", y.shape)

    # 计算总数据长度
    total_len = len(y)
    # print("total_len=", total_len)

    # 划分训练集和测试集，这里以90%数据作为训练集，剩余10%为测试集
    trainx, trainy = x[:int(0.90 * total_len), ], y[:int(0.90 * total_len), ]
    testx, testy = x[int(0.90 * total_len):, ], y[int(0.90 * total_len):, ]

    # 使用自定义的Mydataset类（需要在其他地方定义）来封装数据，并创建DataLoader，这样可以在训练模型时方便地进行批处理
    train_loader = DataLoader(dataset=Mydataset(trainx, trainy), shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(dataset=Mydataset(testx, testy), shuffle=True, batch_size=batch_size)
    # 返回用于数据反标准化的最大值和最小值，以及训练和测试的数据加载器
    return [close_max, close_min, train_loader, test_loader]

# 定义自己的数据集类Mydataset，继承自Dataset
class Mydataset(Dataset):
    # 构造函数，接收输入x和标签y
    def __init__(self, x, y):
        # 将输入x转换为PyTorch张量
        self.x = torch.from_numpy(x)
        # 将标签y转换为PyTorch张量
        self.y = torch.from_numpy(y)

    # 通过索引获取单个样本，包括输入和标签
    def __getitem__(self, index):
        # 通过索引获取输入x的一个样本
        x1 = self.x[index]
        # 通过索引获取标签y的一个样本
        y1 = self.y[index]
        # 返回获取的样本和标签
        return x1, y1

    # 获取数据集的总长度，即样本数量
    def __len__(self):
        # 返回输入x的长度，也就是样本数量
        return len(self.x)


