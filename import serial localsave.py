import serial
import csv
import os
import pickle
import numpy as np

# 1. 获取唯一文件名
def get_unique_filename(base_name="data", extension=".csv", folder="output"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    counter = 1
    filename = os.path.join(folder, f"{base_name}{extension}")
    while os.path.exists(filename):
        filename = os.path.join(folder, f"{base_name}{counter}{extension}")
        counter += 1
    return filename

# 2. 加载已保存的六个模型
def load_models():
    models = {}
    model_names = ["random_forest1", "random_forest2", "random_forest3", "random_forest4", "random_forest5", "random_forest6"]
    for model_name in model_names:
        model_path = f"C:/Users/ILENS/Desktop/iot/model/{model_name}.pkl" 
        with open(model_path, 'rb') as f:
            models[model_name] = pickle.load(f)
    return models

# 3. 串口设置
ser = serial.Serial('COM8', 9600, timeout=1)
ser.flush()

# 获取唯一文件名并指定存储路径
csv_filename = get_unique_filename(base_name="data", extension=".csv", folder="C:/Users/ILENS/Desktop/iot/data")

# 加载模型
models = load_models()

# 定义模型与睡姿的对应关系
model_to_sleep_posture = {
    "random_forest1": "仰睡",
    "random_forest2": "左单腿",
    "random_forest3": "左双腿",
    "random_forest4": "右单腿",
    "random_forest5": "右双腿",
    "random_forest6": "趴睡"
}

# 4. 开始接收数据并分类
try:
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['Value', 'FSR0', 'FSR1', 'FSR2', 'FSR3', '最符合的分类睡姿', '1得分', '2得分', '3得分', '4得分', '5得分', '6得分'])
        print(f"数据将保存到: {csv_filename}")
        print("正在接收数据，按 Ctrl+C 结束...")

        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                data = line.split(",")
                if len(data) == 5:  
                    input_data = np.array([int(val) for val in data]).reshape(1, -1)
                    scores = []
                    max_score = 0
                    best_label = '其他睡姿'
                    for i, (model_name, model) in enumerate(models.items()):
                        score = model.predict_proba(input_data)[0][1]  
                        scores.append(score)

                        if score > max_score and score > 0.7:
                            max_score = score
                            best_label = model_to_sleep_posture[model_name] 


                    if max_score <= 0.7:
                        best_label = '其他睡姿'


                    writer.writerow(data + [best_label] + scores)
                    print(f"数据: {data}, 最符合的分类睡姿: {best_label}, 得分: {scores}, 最大得分: {max_score:.4f}")

except KeyboardInterrupt:
    print("数据接收已结束")

finally:
    ser.close()
    print(f"数据已保存到: {csv_filename}")
