import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# 1. 加载数据
file_path = "C:/Users/ILENS/Desktop/iot/data/训练数据6趴睡.csv"
data = pd.read_csv(file_path, encoding='utf-8')  
data.columns = data.columns.str.strip()  

# 2. 分离特征和标签
X = data[['Value', 'FSR0', 'FSR1', 'FSR2', 'FSR3']]  
y = data['LABEL']  # 标签

# 3. 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=90)

# 4. 创建随机森林模型，修改 max_features 参数
rf_classifier = RandomForestClassifier(
    n_estimators=8,  
    max_depth=9,     
    max_features= 'sqrt', 
    random_state=90    
)

# 5. 训练模型
rf_classifier.fit(X_train, y_train)

# 6. 评估模型
train_accuracy = rf_classifier.score(X_train, y_train)
test_accuracy = rf_classifier.score(X_test, y_test)
print(f"训练集准确率: {train_accuracy:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")

# 7. 保存训练好的模型
save_path = 'C:/Users/ILENS/Desktop/iot/model/random_forest6.pkl'  
with open(save_path, 'wb') as file:
    pickle.dump(rf_classifier, file)

print(f"模型已保存到: {save_path}")
