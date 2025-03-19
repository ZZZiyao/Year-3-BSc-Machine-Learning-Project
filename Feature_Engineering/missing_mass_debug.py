#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_missing_mass(df):
    M_B0 = 5279.63  

    # 计算修正后的 B0 轴动量
    p_z = df['B0_PZ'] * M_B0 / df['B0_M']


    r_vector = df[['B0_ENDVERTEX_X', 'B0_ENDVERTEX_Y', 'B0_ENDVERTEX_Z']].values - \
               df[['B0_OWNPV_X', 'B0_OWNPV_Y', 'B0_OWNPV_Z']].values
    
    df[['rvector_X', 'rvector_Y', 'rvector_Z']] = df[['B0_ENDVERTEX_X', 'B0_ENDVERTEX_Y', 'B0_ENDVERTEX_Z']].values - \
                                             df[['B0_OWNPV_X', 'B0_OWNPV_Y', 'B0_OWNPV_Z']].values
    
    # 计算方向单位向量
    r_norm = np.linalg.norm(r_vector, axis=1, keepdims=True)

    momentum_direction = r_vector / r_norm
    df[['momentum_dir_X', 'momentum_dir_Y', 'momentum_dir_Z']] = df[['rvector_X', 'rvector_Y', 'rvector_Z']].values / r_norm

    p_x = p_z * (momentum_direction[:, 0] / momentum_direction[:, 2])
    p_y = p_z * (momentum_direction[:, 1] / momentum_direction[:, 2])
    p_magnitude = np.sqrt(p_x**2 + p_y**2 + p_z**2)


    df['B0_PX_C'] = p_x
    df['B0_PY_C'] = p_y
    df['B0_PZ_C'] = p_z
    df['B0_P_C'] = p_magnitude

    # 计算 B0_PE_C
    df['B0_PE_C'] = np.sqrt(M_B0**2 + p_magnitude**2)

    #df['B0_PE'] = df['kaon_PE'] + df['pion_PE'] + df['mu_plus_PE']+ df['mu_minus_PE']
    df['B0_PE']=np.sqrt(df['B0_M']**2 + df['B0_P']**2)

    X = (df['B0_PE_C'] - df['B0_PE'])**2 - \
    (df['B0_PX_C'] - df['B0_PX'])**2 - \
    (df['B0_PY_C'] - df['B0_PY'])**2 - \
    (df['B0_PZ_C'] - df['B0_PZ'])**2

    neg_X_count = (X < 0).sum()
    print(f"有 {neg_X_count} 行数据导致负数 sqrt，可能产生 NaN")
    df['checkX']=X
    # 计算 M_missing
    df['M_missing'] = np.sqrt(
        (df['B0_PE_C'] - df['B0_PE'])**2 - 
        (df['B0_PX_C'] - df['B0_PX'])**2 - 
        (df['B0_PY_C'] - df['B0_PY'])**2 - 
        (df['B0_PZ_C'] - df['B0_PZ'])**2
    )

    # 计算 M_missing 产生的 NaN
    nan_M_missing = df['M_missing'].isna().sum()
    print(f"M_missing 计算后的 NaN 数量: {nan_M_missing}")

    return df

#%%
if __name__ == "__main__":


    bg1 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_bg1.csv')
    num_rows1 = bg1.shape[0]
    print(f"bg数据集有 {num_rows1} 行")

    sig1=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_sig1.csv')
    num_rows2 = sig1.shape[0]
    print(f"sig数据集有 {num_rows2} 行")


    bg1 = compute_missing_mass(bg1)
    sig1 = compute_missing_mass(sig1)

#%%

    plt.figure(figsize=(10, 5))
    plt.hist(bg1['M_missing'], bins=500 , alpha=0.7,density=True, label='background',range=(0,10000))
    plt.hist(sig1['M_missing'], bins=500, alpha=0.7,density=True,label='signal',range=(0,10000))

    plt.xlabel("Missing Mass (MeV)")
    plt.ylabel("Counts")
    plt.title("Missing Mass Distribution")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.hist(bg1['checkX'], alpha=0.7,density=True, label='background')
    plt.hist(sig1['checkX'], alpha=0.7,density=True,label='signal')

    plt.xlabel("Missing Mass^2")
    plt.ylabel("Counts")
    plt.title("Checking zeros")
    plt.legend()
    plt.show()



    # plt.figure(figsize=(10, 5))
    # plt.hist(bgdata['Q_square'], bins=500 , alpha=0.7,density=True, label='background',range=(0,1e10))
    # plt.hist(sigdata['Q_square'], bins=500, alpha=0.7,density=True,label='signal',range=(0,1e10))

    # plt.xlabel("Q^2 (MeV)")
    # plt.ylabel("Counts")
    # plt.title("Q^2 Distribution")
    # plt.legend()
    # plt.show()
#%%
print(bg1[['rvector_X', 'rvector_Y', 'rvector_Z']].describe())
print(bg1[['momentum_dir_X', 'momentum_dir_Y', 'momentum_dir_Z']].describe())
plt.hist(bg1['momentum_dir_X'],bins=100,alpha=0.7,label='X')
plt.hist(bg1['momentum_dir_Y'],bins=100,alpha=0.7,label='Y')
plt.hist(bg1['momentum_dir_Z'], bins=100,label='Z')
plt.xlabel('B0 flying direction')
plt.ylabel('Counts')
plt.legend()
# %%
print(bg1['checkX'].describe())
print(sig1['checkX'].describe())
neg_X_ratio = (bg1['checkX'] < 0).mean() * 100
print(f"X 计算为负数的比例: {neg_X_ratio:.2f}%")
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
sns.boxplot(x=bg1['checkX'])
plt.title("Boxplot of checkX (Missing Mass^2)")
plt.show()


# %%
negative_X_df = bg1[bg1['checkX'] < 0]

print("suoyou X 数据的统计信息：")
print(bg1[['B0_PE_C', 'B0_PE', 'B0_PX_C', 'B0_PX', 'B0_PY_C', 'B0_PY', 'B0_PZ_C', 'B0_PZ']].describe())

# %%
import matplotlib.pyplot as plt

plt.hist((bg1['B0_PE_C'] - bg1['B0_PE'])**2, bins=100, alpha=0.7)
plt.xlabel("(B0_PE_C - B0_PE^)2")
plt.ylabel("Counts")
plt.title("Distribution of Energy Difference^2")
plt.show()

# %%
bg1['momentum_diff_squared'] = (bg1['B0_PX_C'] - bg1['B0_PX'])**2 + \
                              (bg1['B0_PY_C'] - bg1['B0_PY'])**2 + \
                              (bg1['B0_PZ_C'] - bg1['B0_PZ'])**2
plt.hist(bg1['momentum_diff_squared'],bins=100)
print(bg1['momentum_diff_squared'].describe())

# %%
extreme_momentum_df = bg1[bg1['momentum_diff_squared'] > 1e11]
print(extreme_momentum_df[['B0_PX_C', 'B0_PX', 'B0_PY_C', 'B0_PY', 'B0_PZ_C', 'B0_PZ']])

# %%
# 选取 X < 0 的样本数据
negative_X_samples = bg1[bg1['checkX'] < 0]

# 选择关注的列
columns_of_interest = ['B0_PE_C', 'B0_PE', 'B0_PX_C', 'B0_PX', 
                       'B0_PY_C', 'B0_PY', 'B0_PZ_C', 'B0_PZ']

# 计算统计信息
negative_X_stats = negative_X_samples[columns_of_interest].describe()

# 显示 X < 0 的样本数量
print(f"Number of samples with X < 0: {len(negative_X_samples)}")

# 打印统计信息
print(negative_X_stats)
#%%
print(negative_X_df['B0_M'].describe())
print(bg1['B0_M'].describe())
plt.hist(negative_X_df['B0_M'],bins=100,alpha=0.7,density=True,label='Data Causing Imaginary Missing Mass')
plt.hist(bg1['B0_M'],bins=100,alpha=0.7,density=True,label='Original Data')
plt.xlabel('Measured B0 Mass (MeV)')
plt.ylabel('density')
plt.legend()
# %%
plt.hist(negative_X_df['B0_PE_C'],bins=100,alpha=0.7,density=True,label='Data Causing Imaginative Missing Mass')
plt.hist(bg1['B0_PE_C'],bins=100,alpha=0.7,density=True,label='Original Data')
plt.xlabel('Corrected B0 Energy (MeV)')
plt.ylabel('density')
plt.legend()

# %%
plt.hist(negative_X_df['B0_PE'],bins=100,alpha=0.7,density=True,label='Data Causing Imaginative Missing Mass')
plt.hist(bg1['B0_PE'],bins=100,alpha=0.7,density=True,label='Original Data')
plt.xlabel('Observed B0 Energy (MeV)')
plt.ylabel('density')
plt.legend()

# %%
plt.hist(bg1['B0_PE_C'],bins=100,alpha=0.7,density=True,label='Corrected',color='green')
plt.hist(bg1['B0_PE'],bins=100,alpha=0.7,density=True,label='Observed')
plt.xlabel('B0 Energy (MeV)')
plt.ylabel('density')
plt.legend()

# %%
plt.hist((negative_X_df['B0_PE_C']-negative_X_df['B0_PE']),bins=100,alpha=0.7,density=True,label='Data Causing Imaginative Missing Mass')
plt.hist((bg1['B0_PE_C']-bg1['B0_PE']),bins=100,alpha=0.7,density=True,label='Original Data')
plt.xlabel('B0 Energy Difference (MeV)')
plt.ylabel('density')
plt.legend()

# %%
bg1['momentum_diff_squared'] = (bg1['B0_PX_C'] - bg1['B0_PX'])**2 + \
                              (bg1['B0_PY_C'] - bg1['B0_PY'])**2 + \
                              (bg1['B0_PZ_C'] - bg1['B0_PZ'])**2

negative_X_df['momentum_diff_squared'] = (negative_X_df['B0_PX_C'] - negative_X_df['B0_PX'])**2 + \
                              (negative_X_df['B0_PY_C'] - negative_X_df['B0_PY'])**2 + \
                              (negative_X_df['B0_PZ_C'] - negative_X_df['B0_PZ'])**2

plt.hist(negative_X_df['momentum_diff_squared'],bins=100,alpha=0.7,density=True,label='Data Causing Imaginative Missing Mass')
plt.hist(bg1['momentum_diff_squared'],bins=100,alpha=0.7,density=True,label='Original Data')
plt.xlabel('B0 Momentum Difference^2 (MeV/c)')
plt.ylabel('density')
plt.legend()

# %%
plt.hist(negative_X_df['B0_PZ'],bins=100,alpha=0.7,density=True,label='Data Causing Imaginative Missing Mass')
plt.hist(bg1['B0_PZ'],bins=100,alpha=0.7,density=True,label='Original Data')
plt.xlabel('B0 Momentum Difference^2 (MeV/c)')
plt.ylabel('density')
plt.legend()


# %%
plt.hist((bg1['B0_PZ_C']-bg1['B0_PZ'])**2,bins=100,alpha=0.7,density=True,label='Orignal PZ diff^2',color='green')
plt.hist((bg1['B0_PE_C']-bg1['B0_PE'])**2,bins=100,alpha=0.7,density=True,label='Original PE diff^2')
plt.hist((negative_X_df['B0_PZ_C']-negative_X_df['B0_PZ'])**2,bins=100,alpha=0.7,density=True,label='Imaginary PZ diff^2')
plt.hist((negative_X_df['B0_PE_C']-negative_X_df['B0_PE'])**2,bins=100,alpha=0.7,density=True,label='Imaginary PE diff^2')

plt.xlabel('B0 Energy difference')
plt.ylabel('density')
plt.legend()

# %%
plt.hist((bg1['checkX']),bins=1000,alpha=0.7,density=True,label='Original',range=(0,max(bg1['checkX'])))
#plt.hist((negative_X_df['checkX']),bins=1000,alpha=0.7,density=True,label='Imaginary',range=(0,max(negative_X_df['checkX'])))

plt.xlabel('momentum difference')
plt.ylabel('density')
plt.legend()
# %%
plt.hist((negative_X_df['checkX']),bins=1000,alpha=0.7,density=True,label='Original')
#plt.hist((negative_X_df['checkX']),bins=1000,alpha=0.7,density=True,label='Imaginary',range=(0,max(negative_X_df['checkX'])))

plt.xlabel('momentum difference')
plt.ylabel('density')
plt.legend()

# %%
print(negative_X_df['checkX'].describe())
print(bg1['checkX'].describe())
# %%
extreme_neg_count = (negative_X_df['checkX'] < -1e7).sum()
total_count = len(negative_X_df)
extreme_neg_ratio = extreme_neg_count / total_count * 100

print(f"极端负值 (< -10^7) 占比: {extreme_neg_ratio:.2f}%")

# %%
