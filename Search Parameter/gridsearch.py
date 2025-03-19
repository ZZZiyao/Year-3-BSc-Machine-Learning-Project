import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import itertools
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from datetime import datetime
import json



data_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\merged_data_small.csv"
data = pd.read_csv(data_path)

#selected_features_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\Selected_features.txt"
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\50features.txt'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\all_feature_importance_filtered.csv'
selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\filtered_feature_2.csv'
#selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\all_feature_importance.csv'
selected_features = pd.read_csv(selected_features_path)['Feature'].tolist()

X = data[selected_features]
y = data['label']  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
neg_count = y_train.value_counts()[0] 
pos_count = y_train.value_counts()[1]  


# 创建保存模型和日志的文件夹
output_dir = "xgboost_experiments"
os.makedirs(output_dir, exist_ok=True)

# 参数搜索空间
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 4, 6, 8],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight':[1, 5, 10, 20],
    'lambda':[0, 0.5, 1, 5]
}

early_stopping_rounds = 30  # 提前停止
scale_pos_weights = np.linspace(1.0, neg_count / pos_count, num=5)  # 遍历 `scale_pos_weight`

# 遍历参数组合
experiment_idx = 0
for spw in scale_pos_weights:
    for values in itertools.product(*param_grid.values()):
        experiment_idx += 1
        params = dict(zip(param_grid.keys(), values))
        params.update({
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'scale_pos_weight': spw,
            'n_jobs': -1,
            'random_state': 42
        })

        print(f"\n🔹 Running Experiment {experiment_idx} with Params: {params}")

        # 训练数据
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        evals = [(dtrain, 'train'), (dval, 'eval')]

        # 训练模型
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,  # 设个较大的上限
            evals=evals,
            early_stopping_rounds=early_stopping_rounds
        )

        # 预测
        y_pred_prob = model.predict(dval)
        y_pred = (y_pred_prob >= 0.5).astype(int)  # 直接用 0.5 阈值

        # 计算评估指标
        auc = roc_auc_score(y_val, y_pred_prob)
        report = classification_report(y_val, y_pred, output_dict=True)

        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{output_dir}/xgb_model_{experiment_idx}_{timestamp}.json"
        log_filename = f"{output_dir}/training_log_{experiment_idx}_{timestamp}.txt"
        report_filename = f"{output_dir}/classification_report_{experiment_idx}_{timestamp}.json"

        # 保存模型
        model.save_model(model_filename)

        # 保存训练日志
        with open(log_filename, "w") as log_file:
            log_file.write(f"Experiment {experiment_idx}\n")
            log_file.write(f"Parameters: {json.dumps(params, indent=4)}\n")
            log_file.write(f"AUC-ROC: {auc:.4f}\n")
            log_file.write(f"Best Iteration: {model.best_iteration}\n")  # 记录最佳迭代轮数

        # 保存分类报告
        with open(report_filename, "w") as report_file:
            json.dump(report, report_file, indent=4)

        print(f"✅ Model saved: {model_filename}")
        print(f"✅ Training log saved: {log_filename}")
        print(f"✅ Classification report saved: {report_filename}")
