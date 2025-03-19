#%%
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd


data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_3data.csv'
data = pd.read_csv(data_path)


#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\filtered_feature_2.csv'
selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\See Importance\permutation_top100_importance.csv'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\mutual_information_scores.csv'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\all_feature_importance.csv'
feature_importance_list = pd.read_csv(selected_features_path)['Feature'].tolist()

X = data[feature_importance_list]
y = data['label']  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



base_params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'seed': 42
}

auc_scores = []
logloss_scores = []
feature_counts = []

for num_features in np.linspace(1, 90, 20, dtype=int): 
    selected_features = feature_importance_list[:num_features]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_val[selected_features]

    model = xgb.XGBClassifier(**base_params)
    model.fit(X_train_selected, y_train)

    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]

    auc = roc_auc_score(y_val, y_pred_proba)
    logloss = log_loss(y_val, y_pred_proba)

    auc_scores.append(auc)
    logloss_scores.append(logloss)
    feature_counts.append(num_features)

    print(f"Num Features: {num_features}, AUC: {auc:.4f}, LogLoss: {logloss:.4f}")
#%%
# 绘制结果
plt.figure(figsize=(10, 5))
plt.plot(feature_counts, auc_scores, marker='o', label='AUC')
plt.plot(feature_counts, logloss_scores, marker='o', label='LogLoss')
plt.axvline(x=43, color='r', linestyle='--', label='Optimal Feature Count (43)')

plt.xlabel('Number of Features')
plt.ylabel('Score')
plt.title('AUC and LogLoss vs Number of Features')
plt.legend()
plt.grid()
plt.show()
# %%
