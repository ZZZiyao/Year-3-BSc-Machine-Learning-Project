#%% 导入必要库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import seaborn as sns

# 1. 读取数据
data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_3data.csv'
data = pd.read_csv(data_path)

# 2. 读取特征
selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\See Importance\permutation_top100_importance.csv'

#selected_features_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\permu_lowcorr_importance.csv'
feature_importance_list = pd.read_csv(selected_features_path)['Feature'].head(43).tolist()

X = data[feature_importance_list]
y = data['label']

# 3. 数据集划分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. 固定 `subsample` 和 `colsample_bytree`
fixed_params = {
    'max_depth': 8,
    'learning_rate': 0.07,
    'subsample': 1.0,  # 固定 subsample
    'colsample_bytree': 1.0,  # 固定 colsample_bytree
    'reg_lambda': 0.1,  # 已优化的 L2 正则化
    'reg_alpha': 10,  # 已优化的 L1 正则化
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'seed': 42
}

# 5. 设定 `gamma` 和 `min_child_weight` 的范围
gamma_values = [0.01, 0.1, 1, 5, 10]  # 横轴 (剪枝参数)
min_child_weight_values = [1, 3, 5, 7, 10]  # 纵轴 (叶子节点最小权重)

# 6. 存储 AUC 分数
heatmap_data = np.zeros((len(min_child_weight_values), len(gamma_values)))

# 7. 遍历 `gamma` 和 `min_child_weight`
for i, gamma in enumerate(gamma_values):
    for j, min_child_weight in enumerate(min_child_weight_values):
        print(f"Testing gamma={gamma}, min_child_weight={min_child_weight}")
        params = fixed_params.copy()
        params['gamma'] = gamma
        params['min_child_weight'] = min_child_weight

        # 转换数据格式
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_val, label=y_val)

        # 训练 XGBoost
        model = xgb.train(params, dtrain, 
                          num_boost_round=500, early_stopping_rounds=30, 
                          evals=[(dtest,'eval')],
                          verbose_eval=False)

        # 计算 AUC
        y_pred = model.predict(dtest)
        auc_score = roc_auc_score(y_val, y_pred)

        # 存储 AUC 分数
        heatmap_data[j, i] = auc_score  # 注意行列索引！

#%% 8. 创建 DataFrame 方便绘图
df = pd.DataFrame(heatmap_data, index=min_child_weight_values, columns=gamma_values)

#%% 9. 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap="viridis", fmt=".5f")
plt.xlabel("Gamma (Pruning Parameter)")
plt.ylabel("Min_child_weight (Leaf Node Min Weight)")
plt.title("Effect of Gamma and Min_child_weight on XGBoost Performance (AUC)")
plt.show()

#%% 10. 绘制折线图
plt.figure(figsize=(10, 6))

# 遍历不同的 `gamma`（不同曲线）
for i, gamma in enumerate(gamma_values):
    plt.plot(min_child_weight_values, heatmap_data[:, i], marker='o', label=f"gamma={gamma}")

# 11. 添加 `gamma=1, min_child_weight=5` 的最佳 AUC 参考线
idx_gamma = gamma_values.index(0.01)
idx_min_child_weight = min_child_weight_values.index(1)
y_value = heatmap_data[idx_min_child_weight, idx_gamma]  # 取最佳 AUC

plt.axhline(y=y_value, color='red', linestyle='--', label=f"optimized: gamma=0.01, min_child_weight=1")
plt.axvline(x=1, color='red', linestyle='--')

# 12. 设置图表属性
plt.xlabel("Min_child_weight (Leaf Node Min Weight)")
plt.ylabel("AUC Score")
plt.title("Effect of Min_child_weight on AUC for Different Gamma Values")
plt.legend(title="Gamma")
plt.grid(True)

# 13. 显示折线图
plt.show()

# %%
