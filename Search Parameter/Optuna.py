import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import shap

data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_3data.csv'
data = pd.read_csv(data_path)

selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\See Importance\permutation_top100_importance.csv'
selected_features = pd.read_csv(selected_features_path)['Feature'].head(52).tolist()

X = data[selected_features]
y = data['label']  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'eta': trial.suggest_loguniform('eta', 0.01, 0.2),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1),
        'lambda': trial.suggest_loguniform('lambda', 0.01, 10),
        'alpha': trial.suggest_loguniform('alpha', 0.01, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'gamma': trial.suggest_loguniform('gamma', 0.01, 10),
        'n_jobs': -1
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(params, dtrain, num_boost_round=500, early_stopping_rounds=30, 
                      evals=[(dtrain, 'train'), (dtest, 'eval')], verbose_eval=False)

    y_train_pred = model.predict(dtrain)
    y_val_pred = model.predict(dtest)

    train_auc = roc_auc_score(y_train, y_train_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)

    auc_gap = train_auc - val_auc

    print(f"Trial {trial.number}: train AUC={train_auc:.5f}, val AUC={val_auc:.5f}, gap={auc_gap:.5f}, Params={params}")

    #I penalize those overfittings
    if auc_gap > 0.03:
        return val_auc - (auc_gap * 5)  
    else:
        return val_auc  

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("best para:", study.best_params)
