import serial
import pickle
import numpy as np
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import time

# MongoDB 配置
MONGO_URI = "mongodb+srv://ilens:xxxx@sleeping.trfss.mongodb.net/mongodbVSCodePlaygroundDB?retryWrites=true&w=majority"
DB_NAME = "mongodbVSCodePlaygroundDB"
COLLECTION_NAME = "final"

# 特征列名称 (与模型训练时一致)
FEATURE_COLUMNS = ["Value", "FSR0", "FSR1", "FSR2", "FSR3"]

# 定义不良睡姿的集合
BAD_POSTURES = {"Left single leg", "Left double leg", "Right single leg", "Right double leg", "On stomach"}

# 加载已保存的六个模型
def load_models():
    models = {}
    model_names = ["random_forest1", "random_forest2", "random_forest3", "random_forest4", "random_forest5", "random_forest6"]
    for model_name in model_names:
        model_path = f"C:/Users/ILENS/Desktop/iot/model/{model_name}.pkl"  
        with open(model_path, 'rb') as f:
            models[model_name] = pickle.load(f)
    return models

# 连接 MongoDB 数据库
def connect_to_mongodb():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return collection

# 定义串口
ser = serial.Serial('COM8', 9600, timeout=1)
ser.flush()

# 加载模型和数据库
models = load_models()
collection = connect_to_mongodb()

# 定义模型与睡姿的对应关系
model_to_sleep_posture = {
    "random_forest1": "On back",
    "random_forest2": "Left single leg",
    "random_forest3": "Left double leg",
    "random_forest4": "Right single leg",
    "random_forest5": "Right double leg",
    "random_forest6": "On stomach"
}

# 定义全局变量
bad_posture_start_time = None  
alert_sent = False  

def process_and_upload_data():
    global bad_posture_start_time, alert_sent
    latest_record = None 

    while True: 
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            data = line.split(",")
            if len(data) == 5: 
                try:
                    
                    input_data = pd.DataFrame([data], columns=FEATURE_COLUMNS).astype(int)

                    scores = []
                    max_score = 0
                    best_label = "Other"

                    for model_name, model in models.items():
                        score = model.predict_proba(input_data)[0][1]
                        scores.append(score)
                        if score > max_score and score > 0.7:
                            max_score = score
                            best_label = model_to_sleep_posture[model_name]

                    if max_score <= 0.7:
                        best_label = "Other"

            
                    if best_label in BAD_POSTURES:
                        if bad_posture_start_time is None:
                            bad_posture_start_time = time.time()
                        elif time.time() - bad_posture_start_time >= 2 and not alert_sent:
                            
                            ser.write(b"ALERT\n")
                            print("不良睡姿超过1分钟，已发送警报信号")
                            alert_sent = True
                    else:
                       
                        bad_posture_start_time = None
                        alert_sent = False

                
                    latest_record = {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Value": int(data[0]),
                        "FSR0": int(data[1]),
                        "FSR1": int(data[2]),
                        "FSR2": int(data[3]),
                        "FSR3": int(data[4]),
                        "Sleeping posture": best_label,
                        "Accuracy value": max_score, 
                        "Scores": scores
                    }

                except Exception as e:
                    print(f"数据处理错误: {e}")

 
    if latest_record:
        try:
            collection.insert_one(latest_record)
            print("已上传当前帧到 MongoDB:", latest_record)
        except Exception as e:
            print(f"MongoDB 上传错误: {e}")
    else:
        print("未收集到有效数据，未上传到 MongoDB")


try:
    print("程序启动，按 Ctrl+C 结束...")
    process_and_upload_data()

except KeyboardInterrupt:
    print("程序已结束")

finally:
    ser.close()
    print("串口已关闭")
