import os
from flask_cors import CORS
import matplotlib
from flask import Flask, jsonify, request

from main import f1, f2, getNewAQI, get_aqi_level

app = Flask(__name__)
CORS(app)

# Matplotlib绘图库使用 "TkAgg" 后端来渲染图形
matplotlib.use('Agg')

# 数据集文件的路径
root = "/Users/admin/src/aqi/file.csv"

# 获取当前脚本文件的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建plot.png的绝对路径
# image_path = os.path.join(script_dir, 'plot.png')

# 获取最近24小时内武汉市空气质量情况
url = "https://eolink.o.apispace.com/34324/air/v001/pastaqi"
payload = {"areacode": "101200101"}
headers = {
    "X-APISpace-Token": "pj4qjwd7umjrukymjwx53fbm4zm0ogh0"
}
# pj4qjwd7umjrukymjwx53fbm4zm0ogh0
# 数据存储路径
now_all_root = "/Users/admin/src/aqi/now_all.csv"
now_use_root = "/Users/admin/src/aqi/now_use.csv"
model_path1 = '/Users/admin/src/aqi/RNN_state_dict.pth'
model_path2 = '/Users/admin/src/aqi/LSTM_state_dict.pth'
model_path3 = '/Users/admin/src/aqi/GRU_state_dict.pth'


model_paths = {
    'model1': model_path1,
    'model2': model_path2,
    'model3': model_path3,
}


# 定义一个简单的路由：根路径
@app.route('/digital')  # http://127.0.0.1:5000/digital?model=model3
def hello_world():
    model = request.args.get('model', 'default_model_id')
    new_df = getNewAQI(url, payload, headers)
    data = f1(new_df, model_paths[model])
    level = get_aqi_level(data[1])
    # 构造响应数据，直接使用转换后的列表
    response_data = {
        "received": data[1],
        "level":level,
        "message": "ok"
    }

    return jsonify(response_data)


# 定义一个POST请求的路由，用于处理数据
@app.route('/image')  # http://127.0.0.1:5000/image?model=model3
def process_data():
    model = request.args.get('model', 'default_model_id')
    new_df = getNewAQI(url, payload, headers)
    image_path = os.path.join(script_dir, model+'plot.png')
    mse, r2, ac, mae = f2(new_df, model_paths[model], image_path, now_all_root, now_use_root)

    # 构造响应数据，直接使用转换后的列表
    response_data = {
        "mse": mse,
        "r2": r2,
        "accuracy": ac,
        "mae": mae,
        "message": "ok"
    }

    # 返回JSON响应
    return jsonify(response_data)


# 运行Flask应用
if __name__ == '__main__':
    app.run(debug=True)
