from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt

model_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\modelhi.json"

loaded_model = XGBClassifier()
loaded_model.load_model(model_path)

print("Successfully Loaded Model!")

data_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\merged_test.csv"
data = pd.read_csv(data_path)
#selected_features_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\Selected_features.txt"
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\50features.txt'
selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\filtered_feature_2.csv'

selected_features = pd.read_csv(selected_features_path)['Feature'].tolist()
#selected_features=['kaon_M', 'pion_MC15TuneV1_ProbNNmu', 'pion_ProbNNmu', 'kaon_hasRich', 'kaon_MC15TuneV1_ProbNNmu', 'pion_M', 'kaon_ProbNNmu', 'mu_minus_isMuon', 'mu_plus_isMuon', 'B0_ENDVERTEX_NDOF', 'pion_hasRich', 'mu_minus_hasRich', 'mu_plus_M', 'mu_plus_hasRich', 'mu_minus_M', 'kaon_ProbNNd', 'mu_plus_ProbNNd', 'pion_ProbNNd', 'mu_minus_ProbNNd', 'pion_ID']
X_val = data[selected_features]

y_val = data['label']  


y_pred_prob = loaded_model.predict_proba(X_val)[:, 1]  

import matplotlib.pyplot as plt
import seaborn as sns

# 真实类别为 0 和 1 的样本的预测概率
prob_0 = y_pred_prob[y_val == 0]
prob_1 = y_pred_prob[y_val == 1]

plt.figure(figsize=(8, 5))
sns.histplot(prob_0, bins=50, kde=True, alpha=0.5, label="True Class 0", color='blue')
sns.histplot(prob_1, bins=50, kde=True, alpha=0.5, label="True Class 1", color='red')

plt.xlabel("Predicted Probability of Class 1")
plt.ylabel("Frequency")
plt.title("Predicted Probability Distributions for Class 0 and Class 1")
plt.legend()
plt.show()






y_pred = (y_pred_prob > 0.5).astype(int)  

import numpy as np

unique, counts = np.unique(y_pred, return_counts=True)
print("Predicted Class Distribution:")
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")


comparison_df = pd.DataFrame({'Actual': y_val, 'Predicted': y_pred})


confusion_matrix = pd.crosstab(comparison_df['Actual'], comparison_df['Predicted'], rownames=['Actual'], colnames=['Predicted'])

print("Confusion Matrix:")
print(confusion_matrix)

