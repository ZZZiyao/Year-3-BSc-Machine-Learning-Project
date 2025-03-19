import xgboost as xgb
import pandas as pd

data_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\merged_data_small.csv"
data = pd.read_csv(data_path)

X_train = data.drop(columns=[col for col in data.columns if 'mother' in col.lower()]
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
              +['label', 'eventNumber', 'runNumber','kstar_M','Polarity'])

y_train = data['label']  

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
feature_importance.to_csv("150_feature_importance.csv", index=False)

print(feature_importance)
