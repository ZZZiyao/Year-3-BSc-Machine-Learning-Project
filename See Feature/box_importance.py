import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


save_dir = r"D:\Year3\BSc Project\Particle-Machine-Learning\Ranked_Boxplots"
os.makedirs(save_dir, exist_ok=True)


data_files = [
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg1.csv", 
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg2.csv",
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg3.csv", 
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg4.csv",
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg5.csv", 
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg6.csv",
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig1.csv", 
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig2.csv", 
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig3.csv", 
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig4.csv",
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig5.csv", 
    r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig6.csv"
]
datasets = [pd.read_csv(f) for f in data_files]


importance_df = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\all_feature_importance2.csv')
features = importance_df['Feature'].tolist()

bg_colors = sns.color_palette("Blues", 6)  
sig_colors = sns.color_palette("Reds", 6)  
colors = bg_colors + sig_colors


for df in datasets:
    for col in df.columns:
        if df[col].dtype == bool or df[col].dtype == "object":  
            df[col] = df[col].astype(int)


dataset_labels = [f"BG {i+1}" for i in range(6)] + [f"SIG {i+1}" for i in range(6)]

i=1
for feature in features:
    
    data_melted = pd.concat(
        [df[[feature]].assign(dataset=dataset_labels[j]) for j, df in enumerate(datasets)]
    ).melt(id_vars=["dataset"], var_name="Feature", value_name="Value")

    
    plt.figure(figsize=(10, 6))
    
    
    sns.boxplot(x="Feature", y="Value", hue="dataset", data=data_melted, width=0.6, palette=colors)

    
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Boxplot of {feature}")
    
    
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')

    
    save_path = os.path.join(save_dir, f"{i}rank{feature}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    
    plt.close()
    i+=1

print(f"All boxplots saved to: {save_dir}")

