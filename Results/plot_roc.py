#%%
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score

# 1. 加载模型
model_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\add_model1.pkl"
loaded_model = joblib.load(model_path)
print("✅ Successfully Loaded XGBClassifier Model!")

# 2. 载入数据
data_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\nonbalanced_test_added.csv"
data = pd.read_csv(data_path)

#selected_features_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\See Importance\permutation_top100_importance.csv"
selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\useful_new_features.txt'
selected_features = pd.read_csv(selected_features_path)['Feature'].tolist()

X_val = data[selected_features]
y_val = data['label']

# 3. 预测概率
dtest = xgb.DMatrix(X_val)
y_scores = loaded_model.predict(dtest)  # 直接返回类别 1 的概率


# 4. 计算ROC曲线
fpr, tpr, _ = roc_curve(y_val, y_scores)
roc_auc = auc(fpr, tpr)

# 5. 计算Precision, Recall, Accuracy随阈值的变化
thresholds = np.linspace(0, 1, 100)
precisions, recalls, f1_scores,accuracies = [], [], [],[]

for thresh in thresholds:
    y_pred = (y_scores >= thresh).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary', zero_division=0)
    accuracy = accuracy_score(y_val, y_pred)
    f1_scores.append(f1)

    precisions.append(precision)
    recalls.append(recall)
    accuracies.append(accuracy)
#%%
# 6. 绘图

# 画ROC曲线
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', lw=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
#%%
# 画Precision, Recall, Accuracy随阈值变化
#plt.plot(thresholds, precisions, label='Precision', color='red')
#plt.plot(thresholds, recalls, label='Recall', color='blue')
plt.plot(thresholds, accuracies, label='Accuracy', color='green')
plt.plot(thresholds, f1_scores, label='F1 Score', color='purple')

plt.xlabel('Decision Threshold')
plt.ylabel('Score')
plt.title('Precision, Recall, and Accuracy vs. Threshold')
plt.legend(loc="best")

plt.tight_layout()
plt.show()
# %%
