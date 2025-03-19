import pandas as pd
from sklearn.preprocessing import StandardScaler

# 读取数据
data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_data.csv'
data = pd.read_csv(data_path)

# 读取特征重要性文件，选择特征
selected_features_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\permutation_feature_importance.csv'
selected_features = pd.read_csv(selected_features_path)['Feature'].tolist()
data = data[selected_features]

# 先进行标准化（Z-score），消除尺度影响
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# 计算标准化后的方差
feature_variance = data_scaled.var()

# 设定方差阈值，比如 < 0.01 认为是小方差
low_variance_features = feature_variance[feature_variance < 0.01]

# 打印方差小的特征
print("📊 低方差特征（方差 < 0.01）(标准化后)：")
print(low_variance_features)

# 可选：保存到 CSV 文件
low_variance_features.to_csv("low_variance_features_scaled.csv", header=True)

print(f"✅ 低方差特征已保存至 'low_variance_features_scaled.csv'")
