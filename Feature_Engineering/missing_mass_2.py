#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_missing_mass(df):

    # PDG data
    M_B0 = 5279.63  
    
    # calculate corrected B0 z axis momentum
    p_z = df['B0_PZ']*M_B0/df['B0_M']

    #calculate B0 momentum 
    r_vector = df[['B0_ENDVERTEX_X', 'B0_ENDVERTEX_Y', 'B0_ENDVERTEX_Z']].values-\
    df[['B0_OWNPV_X', 'B0_OWNPV_Y', 'B0_OWNPV_Z']].values

    # Compute direction (normalize the vector)
    r_norm = np.linalg.norm(r_vector, axis=1, keepdims=True)  # Compute magnitude
    momentum_direction = r_vector / r_norm  # Unit vector direction

    p_x = p_z * (momentum_direction[:, 0] / momentum_direction[:, 2])
    p_y = p_z * (momentum_direction[:, 1] / momentum_direction[:, 2])

    # Compute total momentum magnitude
    p_magnitude = np.sqrt(p_x**2 + p_y**2 + p_z**2)
     
    # Store the results
    df['B0_PX_C'] = p_x
    df['B0_PY_C'] = p_y
    df['B0_PZ_C'] = p_z
    df['B0_P_C'] = p_magnitude

    #corrected energy, using energy momentum conservation
    #df['B0_PE'] = df['kaon_PE'] + df['pion_PE'] + df['mu_plus_PE']+ df['mu_minus_PE']
    df['B0_PE_C'] = np.sqrt(M_B0**2 + df['B0_P_C']**2)
    df['B0_PE']=np.sqrt(df['B0_M']**2 + df['B0_P']**2)


    # calculate four momentum of k, p
    df['Q_PE'] = df['kaon_PE'] + df['pion_PE'] 
    df['Q_PX'] = df['kaon_PX'] + df['pion_PX'] 
    df['Q_PY'] = df['kaon_PY'] + df['pion_PY'] 
    df['Q_PZ'] = df['kaon_PZ'] + df['pion_PZ'] 


    # calculate missing mass
    df['M_missing'] = np.sqrt( 
            (df['B0_PE_C'] - df['B0_PE'])**2 - 
            (df['B0_PX_C'] - df['B0_PX'])**2 - 
            (df['B0_PY_C'] - df['B0_PY'])**2 - 
            (df['B0_PZ_C'] - df['B0_PZ'])**2
        )
    
    df['M_missing_square'] =\
            (df['B0_PE_C'] - df['B0_PE'])**2 - \
            (df['B0_PX_C'] - df['B0_PX'])**2 - \
            (df['B0_PY_C'] - df['B0_PY'])**2 - \
            (df['B0_PZ_C'] - df['B0_PZ'])**2
    
    df['Q_square'] = (df['B0_PE_C'] - df['Q_PE'])**2-\
        (df['B0_PX_C'] - df['Q_PX'])**2 - \
    (df['B0_PY_C'] - df['Q_PY'])**2 - (df['B0_PZ_C'] - df['Q_PZ'])**2


    nan_M_missing = df['M_missing'].isna().sum()
    total_values = len(df['M_missing'])
    print(f"M_missing 计算后的 NaN 数量: {nan_M_missing}")
    print(f"Percentage of NaN: {nan_M_missing / total_values:.2%}")  # Formatting as percentage




    return df  

#%%

if __name__ == "__main__":


    bg1 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_bg1.csv')

    sig1=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_sig1.csv')


    bgdata = compute_missing_mass(bg1)
    sigdata = compute_missing_mass(sig1)
#%%

    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['M_missing'], bins=500 , alpha=0.7,density=True, label='background')
    plt.hist(sigdata['M_missing'], bins=500, alpha=0.7,density=True,label='signal')

    plt.xlabel("Missing Mass (MeV)")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Missing Mass Distribution")
    plt.show()
#%%
    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['M_missing_square'], bins=500 , alpha=0.7,density=True, label='background',range=(-1e7,max(bgdata['M_missing_square'])))
    plt.hist(sigdata['M_missing_square'], bins=500, alpha=0.7,density=True,label='signal',range=(-1e7,max(sigdata['M_missing_square'])))

    plt.xlabel("Missing Mass Squared (MeV^2)")
    plt.ylabel("Density")
    plt.title("Missing Mass Squared Distribution")
    plt.legend()
    plt.show()
#%%
    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['Q_square'], bins=500 , alpha=0.7,density=True, label='background',
             range=(0,max(bgdata['Q_square']))
             )
    plt.hist(sigdata['Q_square'], bins=500, alpha=0.7,density=True,label='signal',
             range=(0,max(sigdata['Q_square']))
             )

    plt.xlabel("Q Square (MeV^2)")
    plt.ylabel("Density")
    plt.title("Q Square Distribution")
    plt.legend()
    plt.show()








# %%
