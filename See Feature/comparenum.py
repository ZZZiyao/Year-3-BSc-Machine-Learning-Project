#%%
import pandas as pd

bg_filtered=pd.read_csv(r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_bg1.csv")
sig_filtered=pd.read_csv(r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_sig1.csv")

bg_original=pd.read_pickle(r"D:/Year3/BSc Project/Particle-Machine-Learning/dataset/bd2ddkst_munu_MU_2016_1013_reducedbranches.pkl")
sig_original=pd.read_pickle(r"D:/Year3/BSc Project/Particle-Machine-Learning/dataset/output_kpitautau_2016_md_reducedbranches.pkl")



print(bg_filtered.shape)
print(sig_filtered.shape)
print(bg_original.shape)
print(sig_original.shape)
# %%
bg_mid=pd.read_csv(r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg1.csv")
sig_mid=pd.read_csv(r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig1.csv")
print(bg_mid.shape)
print(sig_mid.shape)

print(302925/1281770)
print(119757/436359)
# %%
