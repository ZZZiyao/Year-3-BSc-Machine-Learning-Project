#%%
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt


shap_data = pd.read_csv(r"D:\Year3\BSc Project\Particle-Machine-Learning\shap_values.csv")

shap_importance = shap_data.abs().mean().sort_values(ascending=False)

print(shap_importance)

shap_importance.to_csv(r"D:\Year3\BSc Project\Particle-Machine-Learning\shap_importance.csv", header=True)


#%%
data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_data.csv'
data = pd.read_csv(data_path)

selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\non_boo_feature_importance.csv'
selected_features = pd.read_csv(selected_features_path)['Feature'].tolist()

X = data[selected_features]
y = data['label']  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\testshap.json"

loaded_model = XGBClassifier()
loaded_model.load_model(model_path)

print("Successfully Loaded Model!")
#%%
y_pred_prob = loaded_model.predict_proba(X_val)[:, 1]

y_true = y_val
y_pred = (y_pred_prob > 0.5).astype(int)

wrong_predictions = y_pred != y_true

X_wrong = shap_data[wrong_predictions]
y_wrong_true = y_true[wrong_predictions]
y_wrong_pred = y_pred[wrong_predictions]


shap_importance_wrong = X_wrong.abs().mean().sort_values(ascending=False)

print(shap_importance_wrong.head(20))


correct_predictions = y_pred == y_true
X_correct = shap_data[correct_predictions]

shap_importance_correct = X_correct.abs().mean()
shap_importance_wrong = X_wrong.abs().mean()

shap_diff = (shap_importance_wrong - shap_importance_correct).sort_values(ascending=False)

print(shap_diff.head(20))





