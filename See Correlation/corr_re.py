import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取相关性矩阵
corr_matrix_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\See Correlation\feature_correlation_matrix.csv'
corr_matrix = pd.read_csv(corr_matrix_path, index_col=0)

# 选定特征
selected_features = [
    "kstar_PZ", "kstar_P", "mu_minus_PX", "mu_minus_AtVtx_PX", 
    "mu_plus_AtVtx_PX", "mu_plus_PX", "mu_minus_P", "mu_minus_PZ",
    "mu_plus_PE", "mu_plus_PZ", "pion_PE", "pion_PZ", "mu_plus_AtVtx_PZ",
    "kaon_P", "kaon_PZ", "kstar_PX", "kaon_PX", "B0_PZ", "B0_P", "B0_PX", 
    "kstar_PY", "B0_PY", "kaon_PY", "mu_minus_AtVtx_PZ", "kstar_ORIVX_CHI2", 
    "B0_ENDVERTEX_CHI2", "mu_plus_P", "mu_minus_PE", "kaon_PE", 
    "mu_plus_PY", "mu_plus_AtVtx_PY", "mu_minus_AtVtx_PY", "mu_minus_PY", 
    "pion_P"
]
# 提取相关性子矩阵
filtered_corr_matrix = corr_matrix.loc[selected_features, selected_features]

# 设置相关性阈值
threshold = 0.95

# 绘制相关性热力图
plt.figure(figsize=(12, 10))
sns.heatmap(
    filtered_corr_matrix, 
    annot=False, 
    cmap='coolwarm', 
    center=0, 
    linewidths=0.5
)
plt.title("Feature Correlation Heatmap")
plt.show()
