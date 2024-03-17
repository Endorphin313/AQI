import torch
import torch.nn as nn
import torch.optim as optim

# 定义GRU模型类，继承自nn.Module
class GRUModel(nn.Module):
    # 构造函数定义了模型初始化时需要的参数
    def __init__(self, input_size=7, hidden_size=32, output_size=1, num_layers=1, dropout=0, batch_first=True):
        # 调用父类的构造函数
        super(GRUModel, self).__init__()
        # 初始化模型参数
        self.hidden_size = hidden_size  # GRU单元的隐藏层大小
        self.input_size = input_size  # 输入数据的特征数量
        self.num_layers = num_layers  # GRU的层数
        self.output_size = output_size  # 模型输出的大小
        self.batch_first = batch_first  # 是否将batch作为输入数据的第一个维度
        # 定义GRU层，设置输入大小、隐藏层大小、层数和是否batch为第一个维度
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          batch_first=self.batch_first)
        # 定义一个线性层，将GRU的输出映射到最终的输出大小
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # 前向传播
        out, _ = self.gru(x, h0)
        out = self.linear(out[:, -1, :])
        return out

# 定义训练函数
def train_GRU(input_size, hidden_size,output_size, num_layers,dropout, batch_first,device,learning_rate,num_epochs,train_loader):
    # 初始化模型对象，并传入定义的参数
    model = GRUModel(input_size=input_size, hidden_size=hidden_size,
                     output_size=output_size, num_layers=num_layers,
                     dropout=dropout,batch_first=batch_first)
    # 将模型转移到指定的设备（GPU或CPU）
    model.to(device)
    # 定义损失函数为均方误差损失，用于回归任务
    criterion = nn.MSELoss()  # 定义损失函数
    # 定义优化器为Adam，用于更新模型参数，设置学习率为0.01
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器
    # 开始训练轮次，总共训练epoch次
    for epoch in range(num_epochs):
        # 遍历训练数据加载器中的所有批次
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # 清除旧的梯度
            outputs = model(inputs)  # 获取模型输出
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
        # 将当前轮次的损失值记录下来
        # totall_loss.append(loss.item())
    return model

