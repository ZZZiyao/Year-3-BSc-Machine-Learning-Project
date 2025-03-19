#%%
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import joblib

# 1. 用 joblib 加载模型
model_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\add_model1.pkl"
loaded_model = joblib.load(model_path)  # 适用于 `XGBClassifier().fit()` 训练的模型

print("✅ Successfully Loaded XGBClassifier Model!")

# 2. 载入数据
import pandas as pd
data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\nonbalanced_test_added.csv'
data = pd.read_csv(data_path)

selected_features_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\useful_new_features.txt'
selected_features = pd.read_csv(selected_features_path)['Feature'].tolist()

X_val = data[selected_features]
y_val = data['label']

dtest = xgb.DMatrix(X_val)
y_pred_prob = loaded_model.predict(dtest)  # 直接返回类别 1 的概率
y_pred = (y_pred_prob > 0.406).astype(int)  # 设定阈值为 0.5，将概率转换为类别

print("✅ Predictions Done! First 5 probabilities:", y_pred_prob[:5])

# 4. 获取散点图数据
scatter_data = pd.DataFrame({
    'B0_M': data['B0_M'],
    'B0_ENDVERTEX_CHI2': data['B0_ENDVERTEX_CHI2'],
    'True_Label': y_val,
    'Pred_Label': y_pred
})

# 保存为 CSV 以便检查
scatter_data.to_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\scatter_data.csv', index=False)

print("✅ Scatter data saved!")
#%%
import pandas as pd
import matplotlib.pyplot as plt

# 读取 scatter_data
scatter_data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\scatter_data.csv'
scatter_data = pd.read_csv(scatter_data_path)

# 创建散点图
plt.figure(figsize=(10, 6))

# 原始类别 0（真实值）
plt.scatter(
    scatter_data.loc[scatter_data['True_Label'] == 0, 'B0_M'],
    scatter_data.loc[scatter_data['True_Label'] == 0, 'B0_ENDVERTEX_CHI2'],
    c='blue', marker='o', label='True 0'
)

# 原始类别 1（真实值）
plt.scatter(
    scatter_data.loc[scatter_data['True_Label'] == 1, 'B0_M'],
    scatter_data.loc[scatter_data['True_Label'] == 1, 'B0_ENDVERTEX_CHI2'],
    c='red', marker='o', label='True 1'
)

# 预测类别 0（模型预测）
plt.scatter(
    scatter_data.loc[scatter_data['Pred_Label'] == 0, 'B0_M'],
    scatter_data.loc[scatter_data['Pred_Label'] == 0, 'B0_ENDVERTEX_CHI2'],
    edgecolors='black', marker='x', label='Pred 0', facecolors='none'
)

# 预测类别 1（模型预测）
plt.scatter(
    scatter_data.loc[scatter_data['Pred_Label'] == 1, 'B0_M'],
    scatter_data.loc[scatter_data['Pred_Label'] == 1, 'B0_ENDVERTEX_CHI2'],
    edgecolors='black', marker='s', label='Pred 1', facecolors='none'
)

# 图例和标签
plt.xlabel('B0_M')
plt.ylabel('B0_ENDVERTEX_CHI2')
plt.title('Scatter Plot of B0_M vs B0_ENDVERTEX_CHI2 with True and Predicted Labels')
plt.legend()
plt.grid(True)

# 显示图表
plt.show()


# %%
