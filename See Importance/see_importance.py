import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

model_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\xgboost_model.json"
loaded_model = xgb.Booster()
loaded_model.load_model(model_path)

importance_cover = loaded_model.get_score(importance_type='cover')  # 'weight', 'gain' , 'cover'
importance_gain = loaded_model.get_score(importance_type='gain')  # 'weight', 'gain' , 'cover'
importance_weight = loaded_model.get_score(importance_type='weight')  # 'weight', 'gain' , 'cover'


all_features = set(importance_weight.keys()) | set(importance_gain.keys()) | set(importance_cover.keys())

#create a dataframe, if no value, replace by 0
importance_df = pd.DataFrame({
    "Feature": list(all_features),
    "Weight": [importance_weight.get(f, 0) for f in all_features],
    "Gain": [importance_gain.get(f, 0) for f in all_features],
    "Cover": [importance_cover.get(f, 0) for f in all_features]
})

#normalization
importance_df["Weight"] /= importance_df["Weight"].max()
importance_df["Gain"] /= importance_df["Gain"].max()
importance_df["Cover"] /= importance_df["Cover"].max()


# choose features in top 20
top_weight = set(importance_df.nlargest(30, "Weight")["Feature"])
top_gain = set(importance_df.nlargest(30, "Gain")["Feature"])
top_cover = set(importance_df.nlargest(30, "Cover")["Feature"])

selected_features = list(top_weight | top_gain | top_cover)


filtered_df = importance_df[importance_df["Feature"].isin(selected_features)].sort_values(by="Gain", ascending=False)

filtered_df.to_csv("feature_importance.csv", index=False)


x = np.arange(len(filtered_df))


plt.figure(figsize=(12, 8))
bar_width = 0.3  


plt.barh(x - bar_width, filtered_df["Weight"], height=bar_width, label="Weight", color="steelblue")


plt.barh(x, filtered_df["Gain"], height=bar_width, label="Gain", color="orange")


plt.barh(x + bar_width, filtered_df["Cover"], height=bar_width, label="Cover", color="green")


plt.yticks(x, filtered_df["Feature"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance (Weight, Gain, Cover)")
plt.legend()


plt.gca().invert_yaxis()


#plt.show()
