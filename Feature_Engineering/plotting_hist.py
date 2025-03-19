#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from add_feature import add_feature

# 设置保存路径
save_dir = r"D:\Year3\BSc Project\Particle-Machine-Learning\Carefull_Hist_Plots"
os.makedirs(save_dir, exist_ok=True)

# 文件路径
csv_file = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_3data_more.csv'

# 读取 CSV 数据
df = pd.read_csv(csv_file)

# 设定标签列
df['label'] = df['label'].astype(int)  # 确保是整数类型

# 分离背景和信号数据
datasets = [df[df['label'] == i] for i in range(2)]  # 0: Background, 1: Signal

#%%
# 选择要绘制的 feature
selected_feature = "tau_PT"  # 替换为你想绘制的特征名称

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
sig_filtered, sig_lower, sig_upper = remove_outliers(datasets[1][selected_feature].dropna())  # SIG1

# 计算 histogram 的范围（取更小的 lower bound 和更大的 upper bound）
hist_lower = min(bg_lower, sig_lower)
hist_upper = max(bg_upper, sig_upper)

# 设定数据集标签
dataset_labels = [f"BG {i+1}" for i in range(6)] + [f"SIG {i+1}" for i in range(6)]

# 定义颜色
bg_colors = sns.color_palette("Blues", 6)
sig_colors = sns.color_palette("Reds", 6)
colors = bg_colors + sig_colors
plt.figure(figsize=(10,6))
plt.hist(datasets[0][selected_feature], bins=500, alpha=0.7, density=True, label="Background",
         range=(0,hist_upper)
         )
plt.hist(datasets[1][selected_feature], bins=500, alpha=0.7, density=True, label="Signal", 
         range=(0,hist_upper)
         )

plt.title(f"Tau Transverse Momentum")
#plt.title(f"Mu+ P in lab frame")

plt.xlabel('Momentum(MeV/c)')
plt.ylabel("Density")
plt.legend(title="Class", loc='upper right')

# 调整布局
plt.tight_layout()
plt.show()
# 保存图像
save_path = os.path.join(save_dir, f"{selected_feature}_filtered.png")
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()

print(f"Plot for {selected_feature} saved to: {save_dir}")

# %%
#%%


plt.hist(datasets[0]['MM_P_tau']+datasets[0]['MP_P_tau'], bins=500, color="blue", alpha=0.5, density=True, label="Background",range=(0,4000))
plt.hist(datasets[1]['MM_P_tau']+datasets[1]['MP_P_tau'], bins=500, color="red", alpha=0.5, density=True, label="Signal", range=(0,4000))

plt.title(f"Jpsi_P in tau rest frame")
plt.xlabel('Momentum(MeV/c^2)')
plt.ylabel("Density")
plt.legend(title="Class", loc='upper right')

# 调整布局
plt.tight_layout()
plt.show()
# 保存图像
save_path = os.path.join(save_dir, f"{selected_feature}_filtered.png")
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()

print(f"Plot for {selected_feature} saved to: {save_dir}")

# %%
