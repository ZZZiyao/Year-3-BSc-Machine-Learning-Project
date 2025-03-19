import pandas as pd
import numpy as np

# 读取相关性矩阵
file_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\Corr\feature_correlation_matrix.csv"
corr_matrix = pd.read_csv(file_path, index_col=0)

# 只考虑上三角矩阵（避免重复组合）
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# 找出相关性等于1的特征对
perfect_corr_pairs = []
for col in upper_triangle.columns:
    for row in upper_triangle.index:
        if upper_triangle.loc[row, col] >= 0.99:
            # 过滤掉包含 'mc15' 和 'atvtx' 的特征
            if 'mc15' not in row.lower() and 'mc15' not in col.lower() and 'atvtx' not in row.lower() and 'atvtx' not in col.lower():

                perfect_corr_pairs.append((row, col, upper_triangle.loc[row, col]))

# 转换成 DataFrame 方便查看
df_corr_1 = pd.DataFrame(perfect_corr_pairs, columns=['Feature 1', 'Feature 2', 'Corr'])

# 打印结果
print(df_corr_1)

# 可选：保存到 CSV 文件
df_corr_1.to_csv(r"D:\Year3\BSc Project\Particle-Machine-Learning\Corr\perfectly_correlated_features.csv", index=False)
