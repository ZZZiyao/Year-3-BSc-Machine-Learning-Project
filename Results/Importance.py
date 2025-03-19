#%%
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# 1. 加载模型
model_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\more_model1.pkl"
loaded_model = joblib.load(model_path)
# 假设你加载的 `Booster` 变量是 `booster_model`
from xgboost import XGBClassifier
new_model = XGBClassifier()
new_model._Booster = loaded_model  # 让 `XGBClassifier` 使用 `Booster`

# # 2. 读取数据
# data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_test_added.csv'
# data = pd.read_csv(data_path)

# selected_features_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\useful_new_features.txt'
# selected_features = pd.read_csv(selected_features_path)['Feature'].tolist()

# X_val = data[selected_features]
# y_val = data['label']

# 获取 XGBoost 计算的特征重要性
feature_importance = loaded_model.get_score(importance_type="gain")

# 转换为 DataFrame 并排序
feature_importance_df = pd.DataFrame(list(feature_importance.items()), columns=["Feature", "Importance"])
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False).head(20)

print(feature_importance_df)

#%%
# 5. 绘制特征重要性
plt.figure(figsize=(10, 10))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color="blue")
plt.xlabel("Importance (Gain)")
plt.ylabel("Features")
plt.title("Feature Importance for New model")
plt.gca().invert_yaxis()  # 最高重要性的特征在上
plt.show()

# %%
