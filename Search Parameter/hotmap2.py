#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification

data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_3data.csv'
data = pd.read_csv(data_path)


selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\See Importance\permutation_top100_importance.csv'

feature_importance_list = pd.read_csv(selected_features_path)['Feature'].head(52).tolist()

X = data[feature_importance_list]
y = data['label']  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

fixed_params = {
    'max_depth': 8,
    'learning_rate': 0.07,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'seed': 42
}

subsample_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  
colsample_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  

heatmap_data = np.zeros((len(colsample_values), len(subsample_values)))

for i, subsample in enumerate(subsample_values):
    for j, colsample_bytree in enumerate(colsample_values):
        print(i,j)
        params = fixed_params.copy()
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample_bytree

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(params, dtrain, 
                          num_boost_round=500, early_stopping_rounds=30, 
                          evals=[(dtest,'eval')],
                          verbose_eval=False)

        y_pred = model.predict(dtest)
        auc_score = roc_auc_score(y_val, y_pred)

        heatmap_data[j, i] = auc_score  
#%%
df = pd.DataFrame(heatmap_data, index=colsample_values, columns=subsample_values)

plt.figure(figsize=(10, 6))
import seaborn as sns
sns.heatmap(df, annot=True, cmap="viridis", fmt=".5f")
plt.xlabel("Subsample")
plt.ylabel("Colsample_bytree")
plt.title("Effect of Subsample and Colsample_bytree on XGBoost Performance (AUC)")
plt.show()

# %%
plt.figure(figsize=(10, 6))

for j, max_depth in enumerate(subsample_values):
    plt.plot(colsample_values, heatmap_data[:, j], marker='o', label=f"subsample={max_depth}")

idx_learning_rate = colsample_values.index(1.0) 
idx_max_depth = subsample_values.index(1.0)  
y_value = heatmap_data[idx_learning_rate, idx_max_depth]  

plt.axhline(y=y_value, color='red', linestyle='--', label=f"optimized: subsmaple=1.0, colsample_bytree=1.0")
plt.axvline(x=1.0, color='red', linestyle='--')

plt.xlabel("Colsample_bytree")
plt.ylabel("AUC Score")
plt.title("Effect of Colsample_bytree on AUC for Subsamples")
plt.legend(title="Max Depth")  
plt.grid(True)

plt.show()

# %%
