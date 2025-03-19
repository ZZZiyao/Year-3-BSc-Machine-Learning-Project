#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_tau(df):

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
    df['B0_PT_C']=np.sqrt((df['B0_PX_C'])**2+(df['B0_PY_C'])**2)

    #corrected energy, using energy momentum conservation
    #df['B0_PE'] = df['kaon_PE'] + df['pion_PE'] + df['mu_plus_PE']+ df['mu_minus_PE']
    df['B0_PE_C'] = np.sqrt(M_B0**2 + df['B0_P_C']**2)
    df['B0_PE']=np.sqrt(df['B0_M']**2 + df['B0_P']**2)


    df['tau_P']=df['B0_P_C']-df['kaon_P']-df['pion_P']
    df['tau_PX']=df['B0_PX_C']-df['kaon_PX']-df['pion_PX']
    df['tau_PY']=df['B0_PY_C']-df['kaon_PY']-df['pion_PY']
    df['tau_PZ']=df['B0_PZ_C']-df['kaon_PZ']-df['pion_PZ']
    df['tau_PE']=df['B0_PE_C']-df['kaon_PE']-df['pion_PE']
    df['tau_PT']=df['B0_PT_C']-df['kaon_PT']-df['pion_PT']


    return df



if __name__ == "__main__":


    bg1 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_bg1.csv')

    sig1=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_sig1.csv')


    bgdata = compute_tau(bg1)
    sigdata = compute_tau(sig1)
#%%

    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['theta_rad'], bins=500 , alpha=0.7,density=True, label='background')
    plt.hist(sigdata['theta_rad'], bins=500, alpha=0.7,density=True,label='signal')

    plt.xlabel("momentum of tau/D (MeV)")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Tital Momentum Distribution")
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


