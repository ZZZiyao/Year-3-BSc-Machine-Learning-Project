#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

#%%
data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_3data.csv'
data = pd.read_csv(data_path)

#selected_features_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\Selected_features.txt"
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\50features.txt'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\all_feature_importance_filtered.csv'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\filtered_feature_2.csv'
selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\See Importance\permutation_top100_importance.csv'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\all_feature_importance.csv'
feature_importance_list = pd.read_csv(selected_features_path)['Feature'].head(52).tolist()

X = data[feature_importance_list]
y = data['label']  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


base_params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'seed': 42
}
max_depth_values = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 横轴
learning_rate_values = [0.01, 0.03, 0.05, 0.07,0.09, 0.11, 0.2, 0.3]  # 纵轴
heatmap_data = np.zeros((len(learning_rate_values), len(max_depth_values)))
for i, learning_rate in enumerate(learning_rate_values):
    print(f"Testing learning_rate={learning_rate}")
    for j, max_depth in enumerate(max_depth_values):
        print(f"Testing max_depth={max_depth}")
        params = base_params.copy()
        params['max_depth'] = max_depth
        params['learning_rate'] = learning_rate
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,  
            evals=[(dtest, 'eval')], 
            early_stopping_rounds=30,  
            verbose_eval=False  
        )
        y_pred = model.predict(dtest)
        auc_score = roc_auc_score(y_val, y_pred)
        heatmap_data[i, j] = auc_score
#%%
df = pd.DataFrame(heatmap_data, index=learning_rate_values, columns=max_depth_values)
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap="viridis", fmt=".4f")  
plt.xlabel("Max_depth")
plt.ylabel("Learning_rate")
plt.title("Effect of Max_depth and Learning_rate on XGBoost Performance (AUC)")
plt.show()
# %%
plt.figure(figsize=(10, 6))
for j, max_depth in enumerate(max_depth_values):
    plt.plot(learning_rate_values, heatmap_data[:, j], marker='o', label=f"max_depth={max_depth}")
idx_learning_rate = learning_rate_values.index(0.07)  
idx_max_depth = max_depth_values.index(8)  
y_value = heatmap_data[idx_learning_rate, idx_max_depth]  
plt.axhline(y=y_value, color='red', linestyle='--', label=f"optimized: learning_rate=0.07, max_depth=8")
plt.axvline(x=0.07, color='red', linestyle='--')
plt.legend()
plt.xlabel("Learning Rate")
plt.ylabel("AUC")
plt.title("Effect of Max_depth and Learning_rate on XGBoost Performance (AUC)")
plt.show()
# %%
