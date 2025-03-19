#%%
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score
import joblib
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import joblib
from xgboost import XGBClassifier

model_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\add_model1.pkl"
loaded_model = joblib.load(model_path)  

print("Successfully Loaded XGBClassifier Model!")

import pandas as pd
data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\nonbalanced_test_added.csv'
data = pd.read_csv(data_path)

selected_features_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\useful_new_features.txt'
selected_features = pd.read_csv(selected_features_path)['Feature'].tolist()

X_val = data[selected_features]
y_val = data['label']

dtest = xgb.DMatrix(X_val)
y_pred_prob = loaded_model.predict(dtest) 

print("Predictions Done! First 5 probabilities:", y_pred_prob[:5])

#%%


thresholds = np.linspace(0.01, 0.99, 100)  
f1_scores = []

for threshold in thresholds:
    y_pred_temp = (y_pred_prob > threshold).astype(int)
    f1 = f1_score(y_val, y_pred_temp)
    f1_scores.append(f1)

best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)
print(f"Best F1-score: {best_f1:.5f} at Threshold: {best_threshold:.3f}")

#%%
plt.figure(figsize=(8, 5))
plt.plot(thresholds, f1_scores, label="F1-score Curve", color='green')
plt.axvline(best_threshold, linestyle="--", color="red", label=f"Best Threshold = {best_threshold:.3f}")
plt.xlabel("Threshold")
plt.ylabel("F1-score")
plt.title("F1-score vs. Threshold")
plt.legend()
plt.show()
#%%
y_pred_best = (y_pred_prob > best_threshold).astype(int)


cm = confusion_matrix(y_val, y_pred_best)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  

plt.figure(figsize=(6, 5))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Normalized Confusion Matrix")
plt.show()

unique, counts = np.unique(y_pred_best, return_counts=True)
print("\nPredicted Class Distribution:")
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")

print("\nConfusion Matrix:")
print(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
#%%
prob_0 = y_pred_prob[y_val == 0]  
prob_1 = y_pred_prob[y_val == 1]  

plt.figure(figsize=(8, 5))
sns.histplot(prob_0, bins=50, kde=True, alpha=0.5, label="True Background", color='blue')
sns.histplot(prob_1, bins=50, kde=True, alpha=0.5, label="True Signal", color='red')

plt.axvline(best_threshold, color="black", linestyle="--", label=f"Best Threshold = {best_threshold:.3f}")

plt.xlabel("Predicted Probability of Signal")
plt.ylabel("Frequency")
plt.title("Predicted Probability Distributions for Background and Signal")
plt.legend()
plt.show()
# %%
