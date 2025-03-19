import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

save_dir = r"D:\Year3\BSc Project\Particle-Machine-Learning\Feature_Histograms_Top30"
os.makedirs(save_dir, exist_ok=True)


data_files = [r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg1.csv", 
              r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg2.csv",

              r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig1.csv", 
              r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig2.csv", 
          ]
datasets = [pd.read_csv(f) for f in data_files]

#features with high importance
importance_df=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\feature_importance.csv')

features =importance_df['Feature'].tolist()



bg_colors = sns.color_palette("Oranges", 2)  
sig_colors = sns.color_palette("Greens", 2)  
colors = bg_colors + sig_colors  

#True/False to int
for df in datasets:
    for col in df.columns:
        if df[col].dtype == bool or df[col].dtype == "object":  
            df[col] = df[col].astype(int)


for feature in features:
    plt.figure(figsize=(8, 6))

    
    for i, df in enumerate(datasets):
        sns.histplot(df[feature], bins=50, kde=False, 
                     color=colors[i], 
                     alpha=0.4, 
                     stat="density", label=f"Dataset {i+1}")

    
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')

    
    save_path = os.path.join(save_dir, f"{feature}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()  

print(f"all histogram saved: {save_dir}")
