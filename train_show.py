# 导入PyTorch的神经网络模块
import torch
import pandas as pd
from GRU import train_GRU
from LSTM import train_LSTM
from RNN import train_RNN
from data import getData
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score,mean_absolute_error
import matplotlib
import sns
import seaborn as sns

def test(model, test_loader):
    model.eval()  # 设置模型为评估模式
    actuals = []
    predictions = []
    a = []
    p = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            actuals.extend(target.numpy())
            predictions.extend(output.numpy())
            a.extend(target.numpy())
            p.extend(output.numpy())
    return actuals, predictions,a, p

def show(actuals, predictions,a, p):
    # 计算评估指标
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    ac = explained_variance_score(a, p)
    mae = mean_absolute_error(actuals, predictions)
    print(f'MSE: {mse}, RMSE: {rmse}, R²: {r2},ACC:{ac},MAE:{mae}')

    # # 绘制预测值与实际值
    # plt.figure(figsize=(10, 6))
    # plt.plot(actuals, label='Actual', alpha=0.7)
    # plt.plot(predictions, label='Prediction', alpha=0.7)
    # plt.title('Actual vs Prediction')
    # plt.xlabel('Sample')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.show()

def HeatMap(root):
    df = pd.read_csv(root)
    # 计算相关系数
    correlation_matrix = df.corr()
    print(correlation_matrix)
    plt.figure(figsize=(12, 10))  # 可以调整图的大小
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Heatmap')
    plt.show()


# 数据集文件的路径
root = "/Users/admin/src/aqi/file.csv"
# Matplotlib绘图库使用 "TkAgg" 后端来渲染图形
matplotlib.use('TkAgg')
# 设置模型的输入大小。这通常对应于特征的数量。
input_size = 7
# 设置隐藏层的大小。这是LSTM单元内部状态的维度。
hidden_size = 32
# 设置LSTM层的数量。多层LSTM可以增加模型的复杂度和学习能力。
num_layers = 2
# 设置模型的输出大小。这通常对应于预测任务的目标数量。
output_size = 1
# 设置dropout比率，用于在LSTM层之间添加dropout以防止过拟合。这里设置为0，意味着不使用dropout。
dropout = 0
# 设置批次是否作为输入数据的第一维度。这会影响数据的排列方式。
batch_first = True
# 设置训练的总轮次。一个epoch等于整个数据集前向和后向传递一次。
epoch = 10
# 设置学习率
learning_rate = 0.01
# 设置设备，根据是否有可用的CUDA（GPU），自动选择使用CPU或GPU进行训练。
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 调用getData函数处理数据，并返回处理后的结果，包括归一化的最大值、最小值、训练数据加载器和测试数据加载器
close_max, close_min, train_loader, test_loader = getData(root, sequence_length=24, batch_size=64)
# 打印归一化的最大值和最小值
print("close_max=,close_min=", close_max, close_min)
# 打印训练数据加载器中批次的数量
print("train_loader=", len(train_loader))
# 打印测试数据加载器中批次的数量
print("test_loader=", len(test_loader))

# 展示热力图
HeatMap(root)

# 调用训练函数开始训练模型
model_rnn = train_RNN(input_size, hidden_size,output_size, num_layers,dropout, batch_first,device,learning_rate,epoch,train_loader)
# 保存模型状态字典
# torch.save(model_rnn, 'RNN_state_dict.pth')
# 展示训练相关结果
actuals_rnn, predictions_rnn,a_rnn, p_rnn=test(model_rnn, test_loader)
show(actuals_rnn, predictions_rnn,a_rnn, p_rnn)

# 调用训练函数开始训练模型
model_lstm = train_LSTM(input_size, hidden_size,output_size, num_layers,dropout, batch_first,device,learning_rate,epoch,train_loader)
# 保存模型状态字典
# torch.save(model_lstm, 'LSTM_state_dict.pth')
# 展示训练相关结果
actuals_lstm, predictions_lstm, a_lstm, p_lstm=test(model_lstm, test_loader)
show(actuals_lstm, predictions_lstm, a_lstm, p_lstm)

# 调用训练函数开始训练模型
model_gru = train_GRU(input_size, hidden_size,output_size, num_layers,dropout, batch_first,device,learning_rate,epoch,train_loader)
# 保存模型状态字典
# torch.save(model_gru, 'GRU_state_dict.pth')
# 展示训练相关结果
actuals_gru, predictions_gru, a_gru, p_gru=test(model_gru, test_loader)
show(actuals_gru, predictions_gru, a_gru, p_gru)

