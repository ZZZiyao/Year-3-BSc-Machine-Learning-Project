import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 读取数据
data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_data.csv'
data = pd.read_csv(data_path, low_memory=False)

# 读取特征重要性文件，选择特征
selected_features_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\permutation_feature_importance.csv'
selected_features = pd.read_csv(selected_features_path)['Feature'].tolist()
data = data[selected_features]


# **删除低方差特征（方差接近 0 的特征）**
low_variance_features = [
    "mu_minus_ProbNNd", "mu_plus_M", "pion_ProbNNd",
    "mu_plus_ProbNNd", "pion_M", "B0_ENDVERTEX_NDOF",
    "kaon_ProbNNd", "kaon_M", "mu_minus_M"
]
data_filtered = data.drop(columns=low_variance_features, errors="ignore")

# **计算相关性矩阵**
corr_matrix = data_filtered.corr().abs()

# **取上三角矩阵（去掉对角线）**
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# **找到高度相关（>0.9）的特征**
high_corr_features = [column for column in upper.columns if any(upper[column] > 0.9)]

print("📊 高度相关特征（> 0.9）：")
for feature in high_corr_features:
    max_corr = upper[feature].max()  # 获取该特征的最高相关性
    related_feature = upper[feature].idxmax()  # 获取与其最高相关的特征
    print(f"❌ {feature} (最高相关性: {max_corr:.4f}，与 {related_feature} 相关)")

# **可选：保存相关性矩阵**
corr_matrix.to_csv("feature_correlation_matrix.csv")
print("✅ 相关性矩阵已保存至 feature_correlation_matrix.csv")
