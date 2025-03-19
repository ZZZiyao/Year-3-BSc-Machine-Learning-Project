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
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'seed': 42
}

# 5. 设定 `reg_lambda` 和 `reg_alpha` 的范围
lambda_values = [0, 0.01, 0.1, 1, 5, 10]  # 横轴 (L2 正则化)
alpha_values = [0, 0.01, 0.1, 1, 5, 10]  # 纵轴 (L1 正则化)

# 6. 存储 AUC 分数
heatmap_data = np.zeros((len(alpha_values), len(lambda_values)))

# 7. 遍历 `reg_lambda` 和 `reg_alpha`
for i, reg_lambda in enumerate(lambda_values):
    for j, reg_alpha in enumerate(alpha_values):
        print(i,j)
        params = fixed_params.copy()
        params['reg_lambda'] = reg_lambda
        params['reg_alpha'] = reg_alpha

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
df = pd.DataFrame(heatmap_data, index=alpha_values, columns=lambda_values)

#%% 9. 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap="viridis", fmt=".5f")
plt.xlabel("Reg_lambda (L2 Regularization)")
plt.ylabel("Reg_alpha (L1 Regularization)")
plt.title("Effect of Reg_lambda and Reg_alpha on XGBoost Performance (AUC)")
plt.show()

#%% 10. 绘制折线图
plt.figure(figsize=(10, 6))

# 遍历不同的 `reg_lambda`（不同曲线）
for i, reg_lambda in enumerate(lambda_values):
    plt.plot(alpha_values, heatmap_data[:, i], marker='o', label=f"reg_lambda={reg_lambda}")

# 11. 添加 `reg_lambda=1, reg_alpha=1` 的最佳 AUC 参考线
idx_lambda = lambda_values.index(0.1)
idx_alpha = alpha_values.index(10)
y_value = heatmap_data[idx_alpha, idx_lambda]  # 取最佳 AUC

plt.axhline(y=y_value, color='red', linestyle='--', label=f"optimized: reg_lambda=0.1, reg_alpha=10")
plt.axvline(x=10, color='red', linestyle='--')

# 12. 设置图表属性
plt.xlabel("Reg_alpha (L1 Regularization)")
plt.ylabel("AUC Score")
plt.title("Effect of Reg_alpha on AUC for Different Reg_lambda Values")
plt.legend(title="Reg Lambda")
plt.grid(True)

# 13. 显示折线图
plt.show()

# %%
