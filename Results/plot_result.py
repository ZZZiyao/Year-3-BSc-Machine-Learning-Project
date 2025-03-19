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

# 1. 用 joblib 加载模型
model_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\new_model2.pkl"
loaded_model = joblib.load(model_path)  # 适用于 `XGBClassifier().fit()` 训练的模型

print("✅ Successfully Loaded XGBClassifier Model!")

# 2. 载入数据
import pandas as pd
data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_test.csv'
data = pd.read_csv(data_path)

selected_features_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\See Importance\permutation_top100_importance.csv'
selected_features = pd.read_csv(selected_features_path)['Feature'].head(52).tolist()

X_val = data[selected_features]
y_val = data['label']

dtest = xgb.DMatrix(X_val)
y_pred_prob = loaded_model.predict(dtest)  # 直接返回类别 1 的概率

print("✅ Predictions Done! First 5 probabilities:", y_pred_prob[:5])

#%%


# 4. 计算 F1-score，找到最佳阈值
thresholds = np.linspace(0.01, 0.99, 100)  # 100 个候选阈值
f1_scores = []

for threshold in thresholds:
    y_pred_temp = (y_pred_prob > threshold).astype(int)
    f1 = f1_score(y_val, y_pred_temp)
    f1_scores.append(f1)

# 找到最大 F1-score 的阈值
best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)
print(f"✅ Best F1-score: {best_f1:.5f} at Threshold: {best_threshold:.3f}")

#%%
# 绘制 F1-score 曲线
plt.figure(figsize=(8, 5))
plt.plot(thresholds, f1_scores, label="F1-score Curve", color='green')
plt.axvline(best_threshold, linestyle="--", color="red", label=f"Best Threshold = {best_threshold:.3f}")
plt.xlabel("Threshold")
plt.ylabel("F1-score")
plt.title("F1-score vs. Threshold")
plt.legend()
plt.show()
#%%
# 6. 计算最佳阈值下的预测结果
y_pred_best = (y_pred_prob > best_threshold).astype(int)

# 7. 计算归一化的混淆矩阵
cm = confusion_matrix(y_val, y_pred_best)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

# 绘制归一化的混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Normalized Confusion Matrix")
plt.show()

# 8. 输出分类分布
unique, counts = np.unique(y_pred_best, return_counts=True)
print("\nPredicted Class Distribution:")
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")

# 9. 输出混淆矩阵（非归一化）
print("\nConfusion Matrix:")
print(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
#%%
# 5. **在 `best_threshold` 下绘制概率分布**
prob_0 = y_pred_prob[y_val == 0]  # 真实类别为 0 的样本的预测概率
prob_1 = y_pred_prob[y_val == 1]  # 真实类别为 1 的样本的预测概率

plt.figure(figsize=(8, 5))
sns.histplot(prob_0, bins=50, kde=True, alpha=0.5, label="True Background", color='blue')
sns.histplot(prob_1, bins=50, kde=True, alpha=0.5, label="True Signal", color='red')

# 在图中标注最佳阈值
plt.axvline(best_threshold, color="black", linestyle="--", label=f"Best Threshold = {best_threshold:.3f}")

plt.xlabel("Predicted Probability of Signal")
plt.ylabel("Frequency")
plt.title("Predicted Probability Distributions for Background and Signal")
plt.legend()
plt.show()
# %%
