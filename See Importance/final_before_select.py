import xgboost as xgb
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


data_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\balanced_3data.csv"
data = pd.read_csv(data_path)


X = data.drop(columns=[col for col in data.columns if 'mother' in col.lower()]
              + [col for col in data.columns if 'true' in col.lower()]
              + [col for col in data.columns if 'mc15' in col.lower()]
              + [col for col in data.columns if 'atvtx' in col.lower()]
              + [col for col in data.columns if 'id' in col.lower()]
              + [col for col in data.columns if data[col].dtype == 'bool']
              + [col for col in data.columns if 'endvertex_x' in col.lower()]
              + [col for col in data.columns if 'endvertex_y' in col.lower()]
              + [col for col in data.columns if 'endvertex_z' in col.lower()]
              + [col for col in data.columns if 'ownpv_x' in col.lower()]
              + [col for col in data.columns if 'ownpv_y' in col.lower()]
              + [col for col in data.columns if 'ownpv_z' in col.lower()]
              + [col for col in data.columns if 'ref' in col.lower()]
              + ['label', 'eventNumber', 'runNumber', 'kstar_M', 'Polarity']
              + ['kstar_PZ', 'mu_minus_PZ', 'mu_plus_PZ', 'pion_PZ', 'kaon_PZ', 'B0_PZ',
                 'mu_plus_PE', 'mu_minus_PE', 'kaon_PE', 'pion_PE', 'nPVs', 'kstar_ORIVX_CHI2']
              + ['mu_minus_ProbNNd', 'mu_plus_M', 'pion_ProbNNd', 'mu_plus_ProbNNd',
                 'pion_M', 'B0_ENDVERTEX_NDOF', 'kaon_ProbNNd', 'kaon_M', 'mu_minus_M'])

y = data['label']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = xgb.XGBClassifier()
model.fit(X_train, y_train)


perm_importance = permutation_importance(model, X_val, y_val, scoring="roc_auc", n_repeats=10, random_state=42)


feature_importance = pd.DataFrame({'Feature': X_val.columns, 'Importance': perm_importance.importances_mean})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)


feature_importance.to_csv("permutation_top100_importance.csv", index=False)

print(feature_importance.head(20))
