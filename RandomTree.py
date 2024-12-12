# from sklearn.datasets import load_breast_cancer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import cross_val_score
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np 

# data= load_breast_cancer()
# rfc = RandomForestClassifier(n_estimators=100, random_state=90)#实例化
# score_pre = cross_val_score(rfc,data.data,data.target,cv=10).mean()
# print("Average score:", score_pre)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

# 1. 加载自定义数据集
file_path = "C:/Users/ILENS/Desktop/iot/data/训练数据6趴睡.csv"
data = pd.read_csv(file_path, encoding='utf-8')  # 加载 CSV 文件
data.columns = data.columns.str.strip()  # 去掉列名的空格

# 2. 分离特征和标签
X = data[['Value', 'FSR0', 'FSR1', 'FSR2', 'FSR3']]  # 输入特征
y = data['LABEL']  # 标签
best_score = 0  # 用于记录最高得分
best_params = {}  # 记录最佳参数
results = []  # 存储所有测试结果

# 3. 实例化随机森林模
# rfc = RandomForestClassifier(n_estimators=100, random_state=90)

# # #==============找最佳n_estimators
# #4. 使用交叉验证评估模型准确性
# score_pre = cross_val_score(rfc, X, y, cv=10).mean()
# print("平均交叉验证准确率:", score_pre)

# # 3. 测试不同 n_estimators 的表现
# scorel = []
# for i in range(1, 101):  # n_estimators 从 1 到 100
#     rfc = RandomForestClassifier(n_estimators=i, random_state=90)
#     score = cross_val_score(rfc, X, y, cv=10).mean()  # 10 折交叉验证
#     scorel.append(score)

# # 4. 找到最优的 n_estimators
# best_score = max(scorel)
# best_n_estimators = scorel.index(best_score) + 1
# print(f"最佳交叉验证得分: {best_score:.4f}, 对应的 n_estimators: {best_n_estimators}")

# # 5. 可视化结果
# plt.figure(figsize=[20, 5])
# plt.plot(range(1, 101), scorel, marker='o')
# plt.xlabel("n_estimators")
# plt.ylabel("Cross-Validation Score")
# plt.title("Random Forest Cross-Validation Accuracy vs. n_estimators")
# plt.show()





#==============最佳max_depth

#3. 手动设置最佳 n_estimators
# best_n_estimators = 8  # 替换为你已经找到的最佳值

# #4. 测试不同 max_depth 的性能
# scores_per_depth = []
# depth_range = range(1, 21)  # max_depth 从 1 到 20
# for depth in depth_range:
#     rfc = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=depth, random_state=90)
#     score = cross_val_score(rfc, X, y, cv=10).mean()  # 10 折交叉验证
#     scores_per_depth.append(score)

# # 5. 找到最佳 max_depth
# best_score = max(scores_per_depth)
# best_max_depth = depth_range[scores_per_depth.index(best_score)]
# print(f"最佳交叉验证得分: {best_score:.4f}, 对应的 max_depth: {best_max_depth}")

# # 6. 可视化 max_depth 的影响
# plt.figure(figsize=[10, 5])
# plt.plot(depth_range, scores_per_depth, marker='o', color='blue')
# plt.xlabel("max_depth")
# plt.ylabel("Cross-Validation Score")
# plt.title(f"Random Forest Accuracy vs. max_depth (n_estimators={best_n_estimators})")
# plt.grid()
# plt.show()

#=============找最佳max_features
#定义参数范围
# param_grid = {'max_features': ['sqrt', 'log2', None, 1, 2, 3, 4, 5]}

# # 实例化随机森林分类器
# rfc = RandomForestClassifier(n_estimators=8, max_depth=9, random_state=90)

# # 5. 使用 GridSearchCV 进行参数搜索
# gs = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10, scoring='accuracy')
# gs.fit(X, y)

# # 6. 输出最佳参数和对应得分
# print("最佳参数:", gs.best_params_)
# print("最佳交叉验证得分:", gs.best_score_)

# # 7. 输出所有参数组合的详细结果
# results = pd.DataFrame(gs.cv_results_)
# print(results[['param_max_features', 'mean_test_score', 'std_test_score']])


#=============学习曲线看测试集和训练集的曲线差距，从而看是否过拟合
# 3. 实例化随机森林模型
rfc = RandomForestClassifier(n_estimators=8, max_depth=9, max_features='sqrt', random_state=90)
#rfc = RandomForestClassifier(n_estimators=16, max_depth=11, random_state=90)

# 4. 生成学习曲线
train_sizes, train_scores, test_scores = learning_curve(
    estimator=rfc,
    X=X,
    y=y,
    cv=10,  # 使用10折交叉验证
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10),  # 不同比例的训练集大小
    random_state=90
)

# 5. 计算训练集和验证集的平均得分及标准差
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# 6. 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

# 绘制误差带
plt.fill_between(train_sizes,
                 train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std,
                 alpha=0.1, color="r")
plt.fill_between(train_sizes,
                 test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std,
                 alpha=0.1, color="g")

# 图表标题和标签
plt.title("Learning Curve: Random Forest")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid()
plt.show()


#==============细致找最佳n_estimators

# scorel = []
# for i in range(1, 201):  # 每次增加 1
#     rfc = RandomForestClassifier(n_estimators=i, random_state=90)
#     score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
#     scorel.append(score)
    
# best_score = max(scorel)
# best_n_estimators = scorel.index(best_score) + 1
# print("Best score:", best_score, "with n_estimators:", best_n_estimators)

# plt.figure(figsize=[20, 5])
# plt.plot(range(1, 201), scorel)
# plt.xlabel("n_estimators")
# plt.ylabel("Cross-Validation Score")
# plt.show()

# scorel = []
# for i in range(60, 80):
#     rfc = RandomForestClassifier(n_estimators=i,
#                                   n_jobs=-1,
#                                   random_state=90)
#     score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
#     scorel.append(score)

# print(max(scorel), ([*range(60, 80)][scorel.index(max(scorel))]))

# plt.figure(figsize=[20, 5])
# plt.plot(range(60, 80), scorel)
# plt.show()

