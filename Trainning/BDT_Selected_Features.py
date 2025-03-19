import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt

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

params = {
    'objective': 'binary:logistic',  
    'eval_metric': ['logloss', 'auc'],  # Evaluation metrics to assess model performance:
                                        # - 'logloss': Measures the quality of predicted probabilities compared to actual labels.
                                        #   A lower value indicates better predictive accuracy.
                                        # - 'auc': Measures the area under the ROC curve, showing the model's ability
                                        #   to separate positive and negative classes. Higher is better (range: 0 to 1).
    'max_depth': 7,                  # Sets the maximum depth of each decision tree:
                                     # - Controls the complexity of the model.
                                     # - Prevents overfitting by limiting tree depth.
                                     # - A value of 5 is a balanced choice for moderately complex data.
    'eta': 0.05,                     # Learning rate (step size shrinkage):
                                     # - Determines how much the weights are updated after each boosting step.
                                     # - A smaller eta value (like 0.05) ensures more gradual learning,
                                     #   making the model less likely to overfit.
                                     # - Lower values typically require more boosting rounds for convergence.
    'subsample': 0.9,                # Row (data) sampling ratio for each tree:
                                     # - Prevents overfitting by training each tree on a random subset of the data.
                                     # - A value of 0.6 means 60% of the training data is randomly sampled for each tree.
                                     # - Lower values make the model more robust, especially for noisy datasets.
    'colsample_bytree': 0.9,         # Column (feature) sampling ratio for each tree:
                                     # - Reduces feature correlation by randomly selecting features for each tree.
                                     # - A value of 0.6 means each tree uses only 60% of the features.
                                     # - Helps to prevent overfitting, especially when the feature set is large.
    'lambda': 1.0,                   # L2 regularization (Ridge regularization):
                                     # - Penalizes large weights in the model to reduce complexity and prevent overfitting.
                                     # - A value of 1.0 is commonly used as a starting point.
                                     # - Higher values (e.g., 2.0 or more) can be tried if the model overfits.
    'alpha': 0.5,                    # L1 regularization (Lasso regularization):
                                     # - Encourages sparsity in the model, effectively selecting important features by
                                     #   driving some feature weights to zero.
                                     # - Useful when the dataset has many features and may contain redundant ones.
                                     # - A value of 0.5 provides moderate regularization strength.
    'min_child_weight': 1,           # Minimum sum of instance weights needed in a leaf node:
                                     # - Prevents the creation of overly deep trees by requiring a minimum number
                                     #   of samples in each leaf node.
                                     # - A value of 1 is suitable for clean and balanced data.
                                     # - Larger values (e.g., 5 or 10) are recommended if the dataset contains noise.
    'max_delta_step': 1,  # 让 Logistic Regression 更稳定
    'n_jobs': -1,  # 使用所有 CPU 核心加速训练


}

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
model.save_model(r"D:\Year3\BSc Project\Particle-Machine-Learning\modelhi.json")



#feature importance
#mother id 411 for bg mother id; mother 15 for signal mother id
#combinatorial; B0 to D+D-kpi