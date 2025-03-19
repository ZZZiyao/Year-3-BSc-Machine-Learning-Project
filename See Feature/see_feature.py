import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

bg1 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg1.csv')
# bg2 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg2.csv')
# bg3 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg3.csv')
# bg4= pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg4.csv')
# bg5= pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg5.csv')
# bg6= pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg6.csv')

sig1=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig1.csv')
# sig2=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig2.csv')
# sig3=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig3.csv')
# sig4=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig4.csv')
# sig5=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig5.csv')
# sig6=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig6.csv')


#print(list(bg.columns))
#print(bg1.shape)
#print(sig1.shape)

k_columns = [col for col in bg1.columns if 'kstar' in col.lower()]

print(k_columns)

bool_columns = [col for col in bg1.columns if bg1[col].dtype == 'bool']
#print(bg1.kaon_MC_GD_GD_MOTHER_ID.unique())

# columns_to_plot = [
#     'mu_minus_MC_GD_GD_MOTHER_ID', 'mu_plus_MC_MOTHER_ID', 'mu_minus_MC_GD_MOTHER_ID',
#     'mu_plus_MC_GD_GD_MOTHER_ID', 'kaon_MC_GD_GD_MOTHER_ID', 'pion_MC_MOTHER_ID',
#     'mu_minus_MC_MOTHER_ID', 'pion_MC_GD_MOTHER_ID', 'mu_plus_MC_GD_MOTHER_ID',
#     'kaon_MC_MOTHER_ID', 'kaon_MC_GD_MOTHER_ID', 'pion_MC_GD_GD_MOTHER_ID'
# ]


# filtered_data = bg[columns_to_plot]


# for col in columns_to_plot:
#     plt.figure(figsize=(8, 5))
#     plt.hist(filtered_data[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
#     plt.xlabel(col)
#     plt.ylabel("Frequency")
#     plt.title(f"Histogram of {col}")
#     plt.grid(axis='y', alpha=0.75)
#     plt.show()

#plt.figure(figsize=(8, 5))
#plt.hist(bg1.mu_minus_MC_GD_GD_MOTHER_ID.dropna(), bins=1000, alpha=1,label='bg1',density=True)
# plt.hist(sig1.Polarity.dropna(), bins=100, alpha=0.7,label='sig1',density=True)
# plt.hist(bg2.Polarity.dropna(), bins=100, alpha=1,label='bg2',density=True)
# plt.hist(sig2.Polarity.dropna(), bins=100, alpha=0.7,label='sig2',density=True)
# plt.hist(bg3.Polarity.dropna(), bins=100, alpha=1,label='bg3',density=True)
# plt.hist(sig3.Polarity.dropna(), bins=100, alpha=0.7,label='sig3',density=True)
# plt.hist(bg4.Polarity.dropna(), bins=100, alpha=1,label='bg4',density=True)
# plt.hist(sig4.Polarity.dropna(), bins=100, alpha=0.7,label='sig4',density=True)
# plt.hist(bg5.Polarity.dropna(), bins=100, alpha=1,label='bg5',density=True)
# plt.hist(sig5.Polarity.dropna(), bins=100, alpha=0.7,label='sig5',density=True)
# plt.hist(bg6.Polarity.dropna(), bins=100, alpha=1,label='bg6',density=True)
# plt.hist(sig6.Polarity.dropna(), bins=100, alpha=0.7,label='sig6',density=True)
# plt.xlabel('Polarity')
# plt.ylabel("Frequency")
# plt.legend()
# plt.title(f"Histogram of kstar_M")
# plt.grid(axis='y', alpha=0.75)
#plt.show()

# plt.figure(figsize=(8, 5))
# plt.hist(bg.Polarity.astype(int).dropna(), bins=2, alpha=1, label='bg', density=True)
# plt.hist(sig.Polarity.astype(int).dropna(), bins=2, alpha=0.7, label='sig', density=True)

# plt.xlabel('3 body decision')
# plt.ylabel("Frequency")
# plt.legend()
# plt.grid(axis='y', alpha=0.75)
# plt.show()