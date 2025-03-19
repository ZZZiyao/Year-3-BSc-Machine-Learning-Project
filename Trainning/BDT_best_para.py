import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import shap

data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_data.csv'
data = pd.read_csv(data_path)

#selected_features_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\Selected_features.txt"
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\50features.txt'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\all_feature_importance_filtered.csv'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\filtered_feature_2.csv'
selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\non_boo_feature_importance.csv'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\all_feature_importance.csv'
selected_features = pd.read_csv(selected_features_path)['Feature'].tolist()

X = data[selected_features]
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
    'max_depth': 10, 
         'eta': 0.07468061198521514, 
         'subsample': 0.953323702936301,
           'colsample_bytree': 0.9740703777079087, 
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
model.save_model(r"D:\Year3\BSc Project\Particle-Machine-Learning\testshap.json")

# Compute SHAP values
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_val)

# Convert SHAP values to DataFrame and save to CSV
shap_df = pd.DataFrame(shap_values.values, columns=selected_features)
shap_df.to_csv(r"D:\Year3\BSc Project\Particle-Machine-Learning\shap_values.csv", index=False)
print("SHAP values saved to CSV.")