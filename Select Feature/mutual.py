#%%
# import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd


data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_3data.csv'
data = pd.read_csv(data_path)

#selected_features_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\Selected_features.txt"
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\50features.txt'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\all_feature_importance_filtered.csv'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\filtered_feature_2.csv'
selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\See Importance\permutation_top100_importance.csv'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\all_feature_importance.csv'
feature_importance_list = pd.read_csv(selected_features_path)['Feature'].tolist()

X = data[feature_importance_list]
y = data['label']  

X_train, _, y_train, _ = train_test_split(X, y, train_size=100000, random_state=42, stratify=y)

mi_scores = mutual_info_classif(X_train, y_train)
#%%
mi_scores_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi_scores})

mi_scores_df = mi_scores_df.sort_values(by='Mutual Information', ascending=False)
output_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\mutual_information_scores.csv'
mi_scores_df.to_csv(output_path, index=False)
#%%
print(mi_scores_df.head(20))

plt.figure(figsize=(10, 6))
plt.barh(mi_scores_df['Feature'][:20], mi_scores_df['Mutual Information'][:20], color='b')
plt.xlabel('Mutual Information')
plt.ylabel('Feature')
plt.title('Top 20 Features by Mutual Information')
plt.gca().invert_yaxis()
plt.show()

# %%
