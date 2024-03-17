import os
import pandas as pd

filepath = "/Users/admin/src/aqi/data"

new_data = []

for r in os.listdir(filepath):
    if r == ".DS_Store":
        continue
    file_path = os.path.join(filepath, r)
    for csv in sorted(os.listdir(file_path)):
        #print(csv)
        count = 0
        csv_path = os.path.join(file_path, csv)
        df = pd.read_csv(csv_path, header=None, sep=',')
        temp = df.iloc[0, :]
        wh_data = []
        for i in temp:
            if i == "武汉":
                wh_data = df.iloc[:, count]
                break
            count += 1
        for i in range(1, len(wh_data), 15):
            n_data = wh_data[i:i+15].values
            n_data = n_data.tolist()
            new_data.append(n_data)

new_df = pd.DataFrame(new_data, columns=['AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h',
                                         'NO2', 'NO2_24h', 'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h'])

# 删除不需要的列
new_df.drop(['PM2.5_24h', 'PM10_24h', 'SO2_24h', 'NO2_24h',  'O3_24h', 'O3_8h', 'O3_8h_24h',  'CO_24h'], axis=1, inplace=True)

# 保存修改后的DataFrame到新的CSV文件
new_file_path = 'modified_file.csv'
new_df.to_csv(new_file_path, index=False)


