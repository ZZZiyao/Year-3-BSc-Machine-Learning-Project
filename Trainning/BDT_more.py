import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import shap

data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_data_with_features.csv'
data = pd.read_csv(data_path)


#selected_features_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\Selected_features.txt"
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\50features.txt'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\all_feature_importance_filtered.csv'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\filtered_feature_2.csv'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\non_boo_feature_importance.csv'
selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\permu_lowcorr_importance.csv'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\all_feature_importance.csv'
selected_features = pd.read_csv(selected_features_path)['Feature'].tolist()

new_features = [
    'M_missing', 'Q_square',
    'delz',  'min_mu_PT',
    'min_mu_PE', 'max_mu_P',
    'max_mu_PZ', 
    'max_mu_ETA', 
    'tau_fd_p', 'tau_fd_m'
    ]


all_features = list(set(selected_features + new_features))

X = data[all_features]
y = data['label']  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#for scale pos weight
neg_count = y_train.value_counts()[0] 
pos_count = y_train.value_counts()[1]  
print(neg_count)
print(pos_count)
print('train set dimension', X_train.shape)
print("test set dimension", X_val.shape)


params= {    'objective': 'binary:logistic',  
    'eval_metric': ['logloss', 'auc'],
    'max_depth': 7, 
         'eta': 0.07468061198521514, 
         'subsample': 0.8,
           'colsample_bytree': 0.8, 
           'lambda': 3.6321468136262696, 
           'alpha': 5.806620604792566, 
           'min_child_weight': 2, 
           'max_delta_step': 9,
           'n_jobs':-1}
#num of trees; 100+trees; validation sample; only keep interested features

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

evals = [(dtrain, 'train'), (dval, 'eval')]

#train the model
model = xgb.train(params, dtrain, num_boost_round=200, evals=evals)

# use test set to predict
y_pred_prob = model.predict(dval)  # probability
# plt.figure(figsize=(8, 5))
# plt.hist(y_pred_prob, bins=50, edgecolor='black', alpha=0.7)
# plt.xlabel("Predicted Probability of Class 1")
# plt.ylabel("Frequency")
# plt.title("Histogram of Predicted Probabilities (y_pred_prob)")
# plt.show()


y_pred = (y_pred_prob > 0.5).astype(int)  # classification

#is 50% a good value?
#how signal-background change when changing the threshold
print("accuracy:", accuracy_score(y_val, y_pred))
print("AUC-ROC:", roc_auc_score(y_val, y_pred_prob))
print("classification report:\n", classification_report(y_val, y_pred))

# save
model.save_model(r"D:\Year3\BSc Project\Particle-Machine-Learning\testmore.json")

# Plot top 30 feature importance
importance_df = pd.DataFrame({'Feature': model.get_score(importance_type='weight').keys(),
                              'Importance': model.get_score(importance_type='weight').values()})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(30)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel("Feature Importance (Weight)")
plt.ylabel("Feature")
plt.title("Top 30 Feature Importance based on XGBoost Importance")
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.savefig(r"D:\Year3\BSc Project\Particle-Machine-Learning\top_30_xgb_importance.png", dpi=300)
plt.show()
print("Top 30 feature importance plot saved.")
