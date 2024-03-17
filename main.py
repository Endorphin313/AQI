import torch
import json
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


def getNewAQI(url, payload, headers):
    response = requests.request("GET", url, params=payload, headers=headers)
    data = json.loads(response.text)
    aqi_data = data['result']['aqis']
    # 转换为Pandas DataFrame
    new_df = pd.DataFrame(aqi_data)
    return new_df


def dataprocess_all(new_df,now_all_root,now_use_root):
    # print("new_all\n")
    # print(new_df)
    old_df = pd.read_csv(now_all_root)
    # print("old\n")
    # print(old_df)
    combined_data = pd.concat([old_df, new_df], ignore_index=True)
    unique_data = combined_data.drop_duplicates(subset=['aqi', 'aqi_level', 'pm10', 'pm25', 'no2', 'so2', 'co', 'o3'], keep='first')
    # print("unique\n")
    # print(unique_data)
    unique_data.to_csv(now_all_root, index=False)
    # 删除不需要的列
    df = unique_data.drop(['aqi_level', 'pollutant', 'data_time'], axis=1)
    # 重命名每列名称
    df.rename(
        columns={'aqi': 'AQI', 'pm10': 'PM10', 'pm25': 'PM2.5', 'no2': 'NO2', 'so2': 'SO2', 'co': 'CO', 'o3': 'O3'},
        inplace=True)
    # 调整每列顺序
    df = df[['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']]
    # print(df)
    df.to_csv(now_use_root, index=False)
    return df


def dataprocess_next(new_df):
    # 删除不需要的列
    df = new_df.drop(['aqi_level', 'pollutant', 'data_time'], axis=1)
    # 重命名每列名称
    df.rename(
        columns={'aqi': 'AQI', 'pm10': 'PM10', 'pm25': 'PM2.5', 'no2': 'NO2', 'so2': 'SO2', 'co': 'CO', 'o3': 'O3'},
        inplace=True)
    # 调整每列顺序
    df = df[['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']]
    return df


def preprocess(input_data, sequence_length=24):
    # 获取预测列的最大值和最小值，用于后续的数据标准化处理
    max = input_data["AQI"].max()
    min = input_data["AQI"].min()

    # 初始化MinMaxScaler，将数据标准化到0和1之间
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 对数据进行标准化处理
    scaled_data = scaler.fit_transform(input_data)

    # 根据sequence_length准备模型输入数据
    data = []
    for i in range(len(scaled_data) - sequence_length + 1):
        data.append(scaled_data[i:i + sequence_length])
    data = np.array(data)

    # 转换为PyTorch张量
    prepared_data = torch.tensor(data, dtype=torch.float32)

    return prepared_data, scaler, max, min


def predict(input_data, model_path):
    # 加载训练好的模型（这里需要替换为实际的模型路径）
    model = torch.load(model_path)
    model.eval()  # 设置为评估模式
    # 使用模型进行预测
    with torch.no_grad():  # 在不计算梯度的情况下进行
        prediction = model(input_data)
    return prediction


def show(df, preds, max, min,image_path):
    buf = BytesIO()
    p = []
    # 将预测值和真实值转换回原始尺度
    for i in range(len(preds)):
        p.append(preds[i][0].item() * (max - min) + min)
    predictions = p[:-1]
    actuals = df.iloc[24:]['AQI'].to_numpy()
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    ac = explained_variance_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    # print(f'MSE: {mse}, R²: {r2},ACC:{ac},MAE:{mae}')

    # 绘制预测值与实际值
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label='Actual', alpha=0.7)
    plt.plot(predictions, label='Prediction', alpha=0.7)
    plt.title('Actual vs Prediction')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    # plt.show()
    plt.savefig(image_path)
    return mse, r2, ac, mae

def next_AQI(df, preds, max, min):
    p = []
    # 将预测值和真实值转换回原始尺度
    for i in range(len(preds)):
        p.append(preds[i][0].item() * (max - min) + min)
    # print(p)
    # predictions = p[:-1]
    # print(predictions)
    # actuals = df.iloc[24:]['AQI'].to_numpy()
    # print(actuals)
    return p

def get_aqi_level(aqi):
    """根据AQI值判断空气质量等级"""
    if aqi <= 50:
        return "优"
    elif aqi <= 100:
        return "良"
    elif aqi <= 150:
        return "轻度污染"
    elif aqi <= 200:
        return "中度污染"
    elif aqi <= 300:
        return "重度污染"
    else:
        return "严重污染"


def f1(new_df, model_path1):
    df_next = dataprocess_next(new_df)
    prepared_data_next, scaler_next, max_next, min_next = preprocess(df_next, 24)
    preds_next = predict(prepared_data_next, model_path1)
    p = next_AQI(df_next,preds_next, max_next, min_next)
    # print("下一小时预测值为：")
    # print(p[1])
    return p


def f2(new_df, model_path1,image_path,now_all_root,now_use_root):
    df_all = dataprocess_all(new_df,now_all_root,now_use_root)
    prepared_data_all, scaler_all, max_all, min_all = preprocess(df_all, 24)
    preds_all = predict(prepared_data_all, model_path1)
    mse, r2, ac, mae = show(df_all,preds_all,max_all, min_all,image_path)
    return mse, r2, ac, mae


