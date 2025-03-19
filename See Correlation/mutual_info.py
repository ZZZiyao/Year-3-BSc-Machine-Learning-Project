from sklearn.feature_selection import mutual_info_classif
import pandas as pd


data_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\merged_data_small.csv"
data = pd.read_csv(data_path)


X = data.drop(columns=[col for col in data.columns if 'mother' in col.lower()]
              + [col for col in data.columns if 'true' in col.lower()]
             + [col for col in data.columns if 'own' in col.lower()]
            + [col for col in data.columns if 'decision' in col.lower()]
            + [col for col in data.columns if 'endvertex_x' in col.lower()]
            + [col for col in data.columns if 'endvertex_y' in col.lower()]
            + [col for col in data.columns if 'endvertex_z' in col.lower()]
              +['label', 'eventNumber', 'runNumber','kstar_M','Polarity'])

y=data['label']

mi_scores = mutual_info_classif(X, y)

# 转换为 Pandas Series 方便查看
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# 选出前 20 个最重要的特征
top_20_mi_features = mi_series.head(20).index.tolist()

print("基于互信息选出的 20 个最重要特征:", top_20_mi_features)
import matplotlib.pyplot as plt
import pandas as pd

# 绘制互信息得分柱状图
plt.figure(figsize=(12, 6))
mi_series.head(20).plot(kind='bar', color='royalblue')

# 设置图表标题和标签
plt.title("Top 20 Features Based on Mutual Information", fontsize=14)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Mutual Information Score", fontsize=12)
plt.xticks(rotation=45, ha='right')  # 旋转 x 轴标签以便更好阅读

# 显示图表
plt.show()
