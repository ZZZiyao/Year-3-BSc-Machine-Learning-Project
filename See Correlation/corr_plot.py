import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 读取相关性矩阵
corr_matrix_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\Corr\feature_correlation_matrix.csv'
corr_matrix = pd.read_csv(corr_matrix_path, index_col=0)

# 选定特征
selected_features = [
    "kstar_PZ", "kstar_P", "mu_minus_PX", "mu_minus_AtVtx_PX", 
    "mu_plus_AtVtx_PX", "mu_plus_PX", "mu_minus_P", "mu_minus_PZ",
    "mu_plus_PE", "mu_plus_PZ", "pion_PE", "pion_PZ", "mu_plus_AtVtx_PZ",
    "kaon_P", "kaon_PZ", "kstar_PX", "kaon_PX", "B0_PZ", "B0_P", "B0_PX", 
    "kstar_PY", "B0_PY", "kaon_PY", "mu_minus_AtVtx_PZ", "kstar_ORIVX_CHI2", 
    "B0_ENDVERTEX_CHI2", "mu_plus_P", "mu_minus_PE", "kaon_PE", 
    "mu_plus_PY", "mu_plus_AtVtx_PY", "mu_minus_AtVtx_PY", "mu_minus_PY", 
    "pion_P"
]

# 提取相关性子矩阵
filtered_corr_matrix = corr_matrix.loc[selected_features, selected_features]

# 设定相关性阈值
threshold = 0.95

# 只保留相关性大于阈值的特征对
high_corr_pairs = filtered_corr_matrix.where(np.triu(np.ones(filtered_corr_matrix.shape), k=0.5).astype(bool))
high_corr_pairs = high_corr_pairs.stack().reset_index()
high_corr_pairs.columns = ["Feature1", "Feature2", "Correlation"]
high_corr_pairs = high_corr_pairs[high_corr_pairs["Correlation"] > threshold]

# 特征名称映射
feature_mapping = {
    "mu_plus": "μ⁺", "mu_minus": "μ⁻", "kaon": "K", "pion": "π", "kstar": "K*", "B0": "B",
    "ENDVERTEX_CHI2": "χ²ₑ", "ORIVX_CHI2": "χ²ₒ", "CHI2": "χ²", "PE": "E",
    "AtVtx": "atV"
}

# 替换特征名称
def rename_feature(feature_name):
    for key, value in feature_mapping.items():
        if key in feature_name:
            feature_name = feature_name.replace(key, value)
    return feature_name

high_corr_pairs["Feature1"] = high_corr_pairs["Feature1"].apply(rename_feature)
high_corr_pairs["Feature2"] = high_corr_pairs["Feature2"].apply(rename_feature)

# **绘制优化后的网络**
def plot_selected_network(corr_pairs, title, filename):
    if corr_pairs.empty:
        print(f"⚠️ 没有找到相关性 > {threshold} 的特征，不绘制图表。")
        return

    G = nx.Graph()
    for _, row in corr_pairs.iterrows():
        G.add_edge(row["Feature1"], row["Feature2"], weight=row["Correlation"])

    # **优化布局**
    plt.figure(figsize=(14, 12))  # 增大画布尺寸
    pos = nx.spring_layout(G, seed=42, k=1.2)  # 调整 k 让节点更分散

    # **节点颜色**
    node_colors = ["royalblue" if "B" in node else 
                   "crimson" if "μ" in node else 
                   "seagreen" for node in G.nodes()]

    # **节点大小**
    degrees = dict(G.degree())
    node_sizes = 800  # 增加节点大小

    # **绘制网络**
    edges = G.edges(data=True)
    edge_weights = [d["weight"] for (_, _, d) in edges]
    edge_widths = [1 + (w - threshold) * 4 for w in edge_weights]

# 绘制节点和边（降低 alpha）
    nx.draw(G, pos, with_labels=False,  # 这里去掉 with_labels
            node_color=node_colors, node_size=node_sizes, 
            edge_color="black", width=edge_widths, 
            alpha=0.6)  # 只降低节点和边的透明度

    # 单独绘制标签，不受 alpha 影响
    nx.draw_networkx_labels(G, pos, font_size=12, 
                            font_color="black", font_weight="bold")

    # **添加边权重**
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_weight="bold")  # 增大边权重字体

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # 调整边距
    plt.title(title, fontsize=18)  # 增大标题字体
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✅ 已保存至 {filename}")

# **绘制优化后的网络**
plot_selected_network(high_corr_pairs, "Selected Feature Correlation Network", 
                      r"D:\Year3\BSc Project\Particle-Machine-Learning\Corr\optimized_correlation_network.png")
