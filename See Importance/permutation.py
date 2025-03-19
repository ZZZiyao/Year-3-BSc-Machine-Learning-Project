import xgboost as xgb
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

data_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\merged_data_small.csv"
data = pd.read_csv(data_path)

#
X = data.drop(columns=[col for col in data.columns if 'mother' in col.lower()]
              + [col for col in data.columns if 'true' in col.lower()]
              + [col for col in data.columns if 'id' in col.lower()]
              + [col for col in data.columns if data[col].dtype == 'bool']
              + [col for col in data.columns if 'endvertex_x' in col.lower()]
              + [col for col in data.columns if 'endvertex_y' in col.lower()]
              + [col for col in data.columns if 'endvertex_z' in col.lower()]
              + [col for col in data.columns if 'ownpv_x' in col.lower()]
              + [col for col in data.columns if 'ownpv_y' in col.lower()]
              + [col for col in data.columns if 'ownpv_z' in col.lower()]
              + [col for col in data.columns if 'ref' in col.lower()]
              + ['label', 'eventNumber', 'runNumber', 'kstar_M', 'Polarity'])

y = data['label']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = xgb.XGBClassifier()
model.fit(X_train, y_train)


perm_importance = permutation_importance(model, X_val, y_val, scoring="roc_auc", n_repeats=10, random_state=42)

feature_importance = pd.DataFrame({'Feature': X_val.columns, 'Importance': perm_importance.importances_mean})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

feature_importance.to_csv("permutation_feature_importance.csv", index=False)


print("📊 Permutation Importance 计算完成，已保存至 'permutation_feature_importance.csv'")
print(feature_importance.head(20))
