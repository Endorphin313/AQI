# 导入PyTorch的神经网络模块
import torch
import torch.nn as nn

# 定义RNN模型类，继承自nn.Module
class RNNModel(nn.Module):
    # 构造函数定义了模型初始化时需要的参数
    def __init__(self, input_size=7, hidden_size=32, num_layers=1, output_size=1, dropout = 0,batch_first=True):
        # 调用父类的构造函数
        super(RNNModel, self).__init__()
        # 初始化模型参数
        self.hidden_size = hidden_size  # RNN单元的隐藏层大小
        self.input_size = input_size    # 输入数据的特征数量
        self.num_layers = num_layers    # RNN的层数
        self.output_size = output_size  # 模型输出的大小
        self.batch_first = batch_first  # 是否将batch作为输入数据的第一个维度
        # 定义RNN层，设置输入大小、隐藏层大小、层数和是否batch为第一个维度
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first)
        # 定义一个线性层，将RNN的输出映射到最终的输出大小
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    # 定义模型的前向传播路径
    def forward(self, x):
        out, hidden = self.rnn(x)
        # 对于多层RNN，hidden的形状为[num_layers, batch, hidden_size]
        # 需要选择最后一层的hidden状态来传递给线性层
        # 如果RNN是单向的，我们可以简单地使用hidden[-1]
        last_hidden = hidden[-1]  # 取最后一层的隐藏状态
        out = self.linear(last_hidden)
        return out

# 定义训练函数
def train_RNN(input_size, hidden_size,output_size, num_layers,dropout, batch_first,device,learning_rate,epoch,train_loader):
    print("RNN：\n")
    # 初始化模型对象，并传入定义的参数
    model = RNNModel(input_size=input_size, hidden_size=hidden_size,
                 output_size=output_size, num_layers=num_layers,
                 dropout=dropout, batch_first=batch_first)
    # 将模型转移到指定的设备（GPU或CPU）
    model.to(device)
    # 定义损失函数为均方误差损失，用于回归任务
    criterion = nn.MSELoss(reduction="mean")
    # 定义优化器为Adam，用于更新模型参数，设置学习率为0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 开始训练轮次，总共训练epoch次
    for i in range(epoch):
        # 遍历训练数据加载器中的所有批次
        for idx, data in enumerate(train_loader, 0):
            # 解包数据，获取输入x和目标y
            x, y = data
            # 将数据转移到指定的设备上
            x = x.to(device)
            y = y.to(device)

            # 对模型进行前向计算，获取预测结果
            pred = model(x)
            # 计算损失值，即预测值与真实值之间的均方误差
            loss = criterion(pred, y)
            # 清零优化器中的梯度，为下一次梯度计算做准备
            optimizer.zero_grad()
            # 对损失值进行反向传播，计算梯度
            loss.backward()
            # 使用优化器更新模型参数
            optimizer.step()
        # # 每训练10轮打印一次当前的轮次和损失值
        # if i % 10 == 0:
        #     print("epoch=", i, "loss=", loss)
        print(f'Epoch {i + 1}/{epoch}, Loss: {loss.item()}')
        # 将当前轮次的损失值记录下来
        # totall_loss.append(loss.item())
    # 返回训练完成的模型
    return model




