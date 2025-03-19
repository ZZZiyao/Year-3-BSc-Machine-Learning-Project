import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt

data_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\merged_data_small.csv"
data = pd.read_csv(data_path)

#selected_features_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\Selected_features.txt"
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\50features.txt'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\all_feature_importance_filtered.csv'
selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\filtered_feature_2.csv'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\all_feature_importance.csv'
selected_features = pd.read_csv(selected_features_path)['Feature'].head(70).tolist()

X = data[selected_features]
y = data['label']  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

import lightgbm as lgb

lgb_clf = lgb.LGBMClassifier(
    objective='binary',
    learning_rate=0.05,
    max_depth=6,
    n_estimators=200,
    subsample=0.7,
    colsample_bytree=0.7,
    lambda_l1=1.0,
    lambda_l2=2.0
)

lgb_clf.fit(X_train, y_train)
y_pred_prob = lgb_clf.predict_proba(X_val)[:, 1]
print("AUC-ROC:", roc_auc_score(y_val, y_pred_prob))
