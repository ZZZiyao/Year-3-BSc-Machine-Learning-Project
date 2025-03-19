import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 文件路径
csv_file = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_3data_more.csv'
features_file = r"D:\Year3\BSc Project\Particle-Machine-Learning\more_feature_names.txt"

# 读取 CSV 数据
df = pd.read_csv(csv_file)

# 读取特征名称
with open(features_file, "r") as f:
    feature_names = [line.strip() for line in f.readlines()]

# 设定标签列
df['label'] = df['label'].astype(int)  # 确保是整数类型

# 分离背景和信号数据
datasets = [df[df['label'] == i] for i in range(2)]  # 0: Background, 1: Signal

# IQR 过滤函数
def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)], lower_bound, upper_bound

# 设置保存目录
save_dir = r"D:\Year3\BSc Project\Particle-Machine-Learning\More_hist_bound"
os.makedirs(save_dir, exist_ok=True)

# 遍历所有特征进行绘图
for selected_feature in feature_names:
    if selected_feature not in df.columns:
        continue
    
    # 过滤异常值并获取区间范围
    _, bg_lower, bg_upper = remove_outliers(datasets[0][selected_feature].dropna())  # Background
    _, sig_lower, sig_upper = remove_outliers(datasets[1][selected_feature].dropna())  # Signal
    
    # 计算 histogram 的范围
    hist_lower = min(bg_lower, sig_lower)
    hist_upper = max(bg_upper, sig_upper)
    
    # 绘制直方图
    plt.figure(figsize=(7, 5))
    plt.hist(datasets[0][selected_feature], bins=500, color="blue", alpha=0.5, density=True, label="Background", 
             range=(hist_lower, hist_upper)
             )
    plt.hist(datasets[1][selected_feature], bins=500, color="red", alpha=0.5, density=True, label="Signal", 
             range=(hist_lower, hist_upper)
             )
    
    plt.title(f"Histogram of {selected_feature}")
    plt.xlabel(selected_feature)
    plt.ylabel("Density")
    plt.legend(title="Class", loc='upper right')
    
    # 保存图像
    save_path = os.path.join(save_dir, f"{selected_feature}_filtered.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Plot for {selected_feature} saved to: {save_path}")
