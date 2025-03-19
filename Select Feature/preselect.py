#%%
import pandas as pd

selected_features_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\See Importance\permutation_top100_importance.csv'

df = pd.read_csv(selected_features_path)

selected_indices = [79, 72, 60, 51, 42, 35, 36, 34, 33, 32, 28, 21, 20, 12, 11, 10, 1]

selected_features = df.iloc[[i - 1 for i in selected_indices], 0]  


print(selected_features.tolist())



#%% Import libraries
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

#%% Load data
data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_3data.csv'
data = pd.read_csv(data_path)

# Load feature importance list
selected_features_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\See Importance\permutation_top100_importance.csv'
feature_importance_df = pd.read_csv(selected_features_path)

# Extract feature names
feature_importance_list = feature_importance_df['Feature'].tolist()

# Indices to remove (1-based index, so we subtract 1)
indices_to_remove = [79, 72, 60, 51, 42, 35, 36, 34, 33, 32, 28, 21, 20, 12, 11, 10, 1]
features_to_remove = [feature_importance_list[i - 1] for i in indices_to_remove]

# Remove selected features
remaining_features = [f for f in feature_importance_list if f not in features_to_remove]

# Split data
X = data[remaining_features]
y = data['label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#%% XGBoost parameters
base_params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'seed': 42
}

#%% Incrementally add features & evaluate performance
auc_scores = []
logloss_scores = []
feature_counts = []

for num_features in np.linspace(1, len(remaining_features), 20, dtype=int): 
    selected_features = remaining_features[:num_features]
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]

    model = xgb.XGBClassifier(**base_params)
    model.fit(X_train_selected, y_train)

    y_pred_proba = model.predict_proba(X_val_selected)[:, 1]

    auc = roc_auc_score(y_val, y_pred_proba)
    logloss = log_loss(y_val, y_pred_proba)

    auc_scores.append(auc)
    logloss_scores.append(logloss)
    feature_counts.append(num_features)

    print(f"Num Features: {num_features}, AUC: {auc:.4f}, LogLoss: {logloss:.4f}")

#%% Plot results
plt.figure(figsize=(10, 5))
plt.plot(feature_counts, auc_scores, marker='o', label='AUC')
plt.plot(feature_counts, logloss_scores, marker='o', label='LogLoss')
plt.axvline(x=52, color='r', linestyle='--', label='Optimal Feature Count (52)')

plt.xlabel('Number of Features')
plt.ylabel('Score')
plt.title('AUC and LogLoss vs Number of Features')
plt.legend()
plt.grid()
plt.show()

# %%
