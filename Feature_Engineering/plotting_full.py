import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from add_feature import add_feature

save_dir = r"D:\Year3\BSc Project\Particle-Machine-Learning\More_Full_Plots"
os.makedirs(save_dir, exist_ok=True)

data_files = [
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_bg1.csv", 
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_bg2.csv",
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_bg3.csv", 
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_bg4.csv",
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_bg5.csv", 
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_bg6.csv",
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_sig1.csv", 
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_sig2.csv", 
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_sig3.csv", 
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_sig4.csv",
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_sig5.csv", 
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_sig6.csv"
]
datas = [pd.read_csv(f) for f in data_files]

datasets = [add_feature(f)[0] for f in datas]  


for file_path, df in zip(data_files, datasets):
    file_name = os.path.basename(file_path)  
    new_file_path = os.path.join(save_dir, f"more_{file_name}")  
    df.to_csv(new_file_path, index=False)  
    print(f"Saved: {new_file_path}")

importance_df = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\more_feature_names.txt')
features = importance_df['Feature'].tolist()


bg_colors = sns.color_palette("Blues", 6)  
sig_colors = sns.color_palette("Reds", 6)  
colors = bg_colors + sig_colors

dataset_labels = [f"BG {i+1}" for i in range(6)] + [f"SIG {i+1}" for i in range(6)]

for feature in features:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    data_melted = pd.concat(
        [df[[feature]].assign(dataset=dataset_labels[j]) for j, df in enumerate(datasets)]
    ).melt(id_vars=["dataset"], var_name="Feature", value_name="Value")

    sns.boxplot(x="dataset", y="Value", data=data_melted, width=0.6, palette=colors, ax=axes[0])
    axes[0].set_title(f"Boxplot of {feature}")
    axes[0].set_xticklabels(dataset_labels, rotation=45, ha="right")

    sns.histplot(datasets[0][feature], bins=1000, kde=False, color="blue", alpha=0.5, stat="density", label="Background", ax=axes[1])
    sns.histplot(datasets[6][feature], bins=1000, kde=False, color="red", alpha=0.5, stat="density", label="Signal", ax=axes[1])

    axes[1].set_title(f"Histogram of {feature}")
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel("Density")
    axes[1].legend(title="Class", loc='upper right')

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{feature}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()  

print(f"All combined plots saved to: {save_dir}")