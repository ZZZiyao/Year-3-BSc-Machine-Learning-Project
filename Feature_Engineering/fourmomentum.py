#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_energy(df):
    df['B0_PE_EM']=np.sqrt(df['B0_M']**2 + df['B0_P']**2)
    df['B0_PE_SUM']=df['kaon_PE'] + df['pion_PE'] + df['mu_plus_PE']+ df['mu_minus_PE']

    return df  



if __name__ == "__main__":


    bg1 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg1.csv')

    sig1=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig1.csv')


    bgdata = compute_energy(bg1)
    sigdata = compute_energy(sig1)


    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['B0_PE_EM'], bins=500 , alpha=0.7,density=True, label='background')
    plt.hist(bgdata['B0_PE_SUM'], bins=500, alpha=0.7,density=True,label='signal')

    plt.xlabel("Missing Mass (MeV)")
    plt.ylabel("Counts")
    plt.title("Missing Mass Distribution")
    plt.legend()
    plt.show()