from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_data.csv'
data = pd.read_csv(data_path)

# 加载已选特征
selected_features_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\permu_lowcorr_importance.csv'
selected_features = pd.read_csv(selected_features_path)['Feature'].tolist()

# 准备特征矩阵和标签
X = data[selected_features]
y = data['label']

# 创建基模型
model = RandomForestClassifier(random_state=42)

# 自定义评分函数
def custom_scorer(estimator, X, y):
    y_pred_proba = estimator.predict_proba(X)[:, 1]  # 预测概率
    auc = roc_auc_score(y, y_pred_proba)  # 计算 AUC
    loss = log_loss(y, y_pred_proba)  # 计算 log loss
    return auc, loss

# 初始化存储结果的列表
auc_scores = []
log_loss_scores = []
num_features_list = []

# 自定义 RFECV 过程
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
selector = RFECV(estimator=model, step=1, cv=cv, scoring='roc_auc')
selector.fit(X, y)

# 在每次迭代中记录 log loss 和 AUC
for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # 使用当前特征子集训练模型
    model.fit(X_train, y_train)
    
    # 计算 AUC 和 log loss
    auc, loss = custom_scorer(model, X_test, y_test)
    
    # 记录结果
    num_features = selector.n_features_  # 当前特征数量
    auc_scores.append(auc)
    log_loss_scores.append(loss)
    num_features_list.append(num_features)
    
    print(f"Iteration {i + 1}: Features = {num_features}, AUC = {auc:.4f}, Log Loss = {loss:.4f}")

# 将结果保存到 DataFrame
results = pd.DataFrame({
    'Num Features': num_features_list,
    'AUC': auc_scores,
    'Log Loss': log_loss_scores
})

# 保存结果到文件
results.to_csv('rfe_cv_results.csv', index=False)

# 绘制 AUC 和 Log Loss 曲线
plt.figure(figsize=(12, 6))

# AUC 曲线
plt.subplot(1, 2, 1)
plt.plot(num_features_list, auc_scores, marker='o', color='b', label='AUC')
plt.xlabel('Number of Features')
plt.ylabel('AUC')
plt.title('AUC vs Number of Features')
plt.grid(True)

# Log Loss 曲线
plt.subplot(1, 2, 2)
plt.plot(num_features_list, log_loss_scores, marker='o', color='r', label='Log Loss')
plt.xlabel('Number of Features')
plt.ylabel('Log Loss')
plt.title('Log Loss vs Number of Features')
plt.grid(True)

plt.tight_layout()
plt.show()