#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from add_feature import add_feature

# 设置保存路径
save_dir = r"D:\Year3\BSc Project\Particle-Machine-Learning\Carefull_New_Plots"
os.makedirs(save_dir, exist_ok=True)

# 读取数据集
data_files = [
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_bg1.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_bg2.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_bg3.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_bg4.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_bg5.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_bg6.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_sig1.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_sig2.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_sig3.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_sig4.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_sig5.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_sig6.csv',

]

datasets = [pd.read_csv(f) for f in data_files]
#%%
# 选择要绘制的 feature
selected_feature = "MET"  # 替换为你想绘制的特征名称

# 计算 IQR 过滤异常值
def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)], lower_bound, upper_bound

# 过滤异常值并获取区间范围
bg_filtered, bg_lower, bg_upper = remove_outliers(datasets[0][selected_feature].dropna())  # BG1
sig_filtered, sig_lower, sig_upper = remove_outliers(datasets[6][selected_feature].dropna())  # SIG1

# 计算 histogram 的范围（取更小的 lower bound 和更大的 upper bound）
hist_lower = min(bg_lower, sig_lower)
hist_upper = max(bg_upper, sig_upper)

# 设定数据集标签
dataset_labels = [f"BG {i+1}" for i in range(6)] + [f"SIG {i+1}" for i in range(6)]

# 定义颜色
bg_colors = sns.color_palette("Blues", 6)
sig_colors = sns.color_palette("Reds", 6)
colors = bg_colors + sig_colors

# 创建绘图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ====== Boxplot (所有 6 个 BG + 6 个 SIG) ======
data_melted = pd.concat(
    [df[[selected_feature]].assign(dataset=dataset_labels[j]) for j, df in enumerate(datasets)]
).melt(id_vars=["dataset"], var_name="Feature", value_name="Value")

sns.boxplot(x="dataset", y="Value", data=data_melted, width=0.6, palette=colors, ax=axes[0])
axes[0].set_title(f"Boxplot of {selected_feature}")
axes[0].set_xticklabels(dataset_labels, rotation=45, ha="right")

# ====== Histogram (仅 BG1 和 SIG1) ======
plt.hist(datasets[0][selected_feature], bins=100, color="blue", alpha=0.5, density=True, label="Background",range=(0,hist_upper))
plt.hist(datasets[6][selected_feature], bins=100, color="red", alpha=0.5, density=True, label="Signal", range=(0,hist_upper))

axes[1].set_title(f"MET")

axes[1].set_xlabel('Momentum(MeV/c^2)')
axes[1].set_ylabel("Density")
axes[1].legend(title="Class", loc='upper right')

# 调整布局
plt.tight_layout()
plt.show()
# 保存图像
save_path = os.path.join(save_dir, f"{selected_feature}_filtered.png")
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()

print(f"Plot for {selected_feature} saved to: {save_dir}")


# %%
