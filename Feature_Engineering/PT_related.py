#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def compute_PT(df):
    df['pT_Kratio'] = df['kaon_PT'] / df['pion_PT']
    df['pT_muratio'] = df['mu_plus_PT'] / df['mu_minus_PT']
    df['pT_mudiff'] = np.abs(df['mu_plus_PT'] - df['mu_minus_PT'])
    df['pT_Kdiff'] = np.abs(df['kaon_PT'] - df['pion_PT'])


    # 计算 MET
    df['MET'] = np.sqrt(df['B0_PX']**2 + df['B0_PY']**2)
    
    # 计算 MET 的方位角 φ_MET
    df['MET_PHI'] = np.arctan2(df['B0_PY'], df['B0_PX'])


    # 计算 Δφ (μ 子与 MET 的方位角差)
    phi1 = np.arctan2(df['mu_plus_PY'], df['mu_plus_PX'])
    phi2 = np.arctan2(df['mu_minus_PY'], df['mu_minus_PX'])
    dphi = np.abs(phi1 - phi2)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)  # Normalize Δφ


    df['DeltaPhi1'] = np.abs(phi1 - df['MET_PHI'])
    df['DeltaPhi1'] = np.where(df['DeltaPhi1'] > np.pi, 2*np.pi - df['DeltaPhi1'], df['DeltaPhi1'])  
    df['DeltaPhi2'] = np.abs(phi2 - df['MET_PHI'])
    df['DeltaPhi2'] = np.where(df['DeltaPhi2'] > np.pi, 2*np.pi - df['DeltaPhi2'], df['DeltaPhi2'])  

    
    # 计算 Δφ (k,pi 子与 MET 的方位角差)
    phik = np.arctan2(df['kaon_PY'], df['pion_PX'])
    phip = np.arctan2(df['kaon_PY'], df['pion_PX'])
    dphikp = np.abs(phik - phip)
    dphikp = np.where(dphikp > np.pi, 2 * np.pi - dphikp, dphikp)  # Normalize Δφ

    df['DeltaPhik'] = np.abs(phik - df['MET_PHI'])
    df['DeltaPhik'] = np.where(df['DeltaPhik'] > np.pi, 2*np.pi - df['DeltaPhik'], df['DeltaPhik'])  
    df['DeltaPhip'] = np.abs(phip - df['MET_PHI'])
    df['DeltaPhip'] = np.where(df['DeltaPhip'] > np.pi, 2*np.pi - df['DeltaPhip'], df['DeltaPhip'])  

    df['mT_mu_plus'] = np.sqrt(2 * df['mu_plus_PT'] * df['MET'] * (1 - np.cos(df['DeltaPhi1'])))
    df['mT_mu_minus'] = np.sqrt(2 * df['mu_minus_PT'] * df['MET'] * (1 - np.cos(df['DeltaPhi2'])))

    return df


if __name__ == "__main__":


    bg1 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_bg1.csv')

    sig1=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_sig1.csv')


    bgdata = compute_PT(bg1)
    sigdata = compute_PT(sig1)

#%%

    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['MET'], bins=500 , alpha=0.7,density=True, label='background')
    plt.hist(sigdata['MET'], bins=500, alpha=0.7,density=True,label='signal')

    plt.xlabel("Missing Mass (MeV)")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Missing Mass Distribution")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['B0_PT'], bins=500 , alpha=0.7,density=True, label='background')
    plt.hist(sigdata['B0_PT'], bins=500, alpha=0.7,density=True,label='signal')

    plt.xlabel("Missing Mass (MeV)")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Missing Mass Distribution")
    plt.show()

#%%
    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['MET_PHI'], bins=500 , alpha=0.7,density=True, label='background')
    plt.hist(sigdata['MET_PHI'], bins=500, alpha=0.7,density=True,label='signal')

    plt.xlabel("Missing Mass Squared (MeV^2)")
    plt.ylabel("Density")
    plt.title("Missing Mass Squared Distribution")
    plt.legend()
    plt.show()
    #%%
    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['pT_Kdiff'], bins=500 , alpha=0.7,density=True, label='background')
    plt.hist(sigdata['pT_Kdiff'], bins=500, alpha=0.7,density=True,label='signal')

    plt.xlabel("Kaon and Pion PT difference (MeV/c)")
    plt.ylabel("Density")
    plt.title("Kstar PT difference Distribution")
    plt.legend()
    plt.show()
    #%%
    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['DeltaPhi2'], bins=500 , alpha=0.7,density=True, label='background')
    plt.hist(sigdata['DeltaPhi2'], bins=500, alpha=0.7,density=True,label='signal')

    plt.xlabel("Angle between MET and Kaon (rad)")
    plt.ylabel("Density")
    plt.title("Angle Distribution")
    plt.legend()
    plt.show()
    #%%
    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['mT_mu_minus'], bins=500 , alpha=0.7,density=True, label='background')
    plt.hist(sigdata['mT_mu_minus'], bins=500, alpha=0.7,density=True,label='signal')

    plt.xlabel("mu minus transverse mass (MeV/c^2)")
    plt.ylabel("Density")
    plt.title("Transverse Mass Distribution")
    plt.legend()
    plt.show()




                                







