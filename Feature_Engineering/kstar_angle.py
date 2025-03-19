#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_kangles(df):
    # Compute phi (azimuthal angle)
    phi1 = np.arctan2(df['kaon_PY'], df['kaon_PX'])
    phi2 = np.arctan2(df['pion_PY'], df['pion_PX'])
    dphi = np.abs(phi1 - phi2)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)  # Normalize Δφ
    df['kdphi']=dphi

    
    eta1 = df['kaon_ETA']
    eta2 = df['pion_ETA']
    deta=np.abs(eta1 - eta2)
    df['kdeta'] = deta

    # Compute ΔR (angular distance)
    df['kdR'] = np.sqrt(deta**2 + dphi**2)
    
    # Compute momentum difference
    p1 = np.sqrt(df['kaon_PX']**2 + df['kaon_PY']**2 + df['kaon_PZ']**2)
    p2 = np.sqrt(df['pion_PX']**2 + df['pion_PY']**2 + df['pion_PZ']**2)
    df['kmomentum_diff'] = np.abs(p1 - p2) / (p1 + p2)

    # Compute angle between trajectories
    dot_product = (df['kaon_PX'] * df['pion_PX'] + 
                   df['kaon_PY'] * df['pion_PY'] + 
                   df['kaon_PZ'] * df['pion_PZ'])
    cos_theta = dot_product / (p1 * p2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Avoid numerical errors
    df['ktheta_rad'] = np.arccos(cos_theta)

    return df


if __name__ == "__main__":


    bg1 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_bg1.csv')

    sig1=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_sig1.csv')


    bgdata = compute_kangles(bg1)
    sigdata = compute_kangles(sig1)

#%%

    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['dphi'], bins=500 , alpha=0.7,density=True, label='background')
    plt.hist(sigdata['dphi'], bins=500, alpha=0.7,density=True,label='signal')

    plt.xlabel("Azimuthal Angle (rad)")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Kstar Angle Distribution")
    plt.show()
#%%
    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['deta'], bins=500 , alpha=0.7,density=True, label='background')
    plt.hist(sigdata['deta'], bins=500, alpha=0.7,density=True,label='signal')

    plt.xlabel("Kstar Pseudorapidity")
    plt.ylabel("Density")
    plt.title("Difference in EAT Distribution")
    plt.legend()
    plt.show()
    #%%
    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['dR'], bins=500 , alpha=0.7,density=True, label='background')
    plt.hist(sigdata['dR'], bins=500, alpha=0.7,density=True,label='signal')

    plt.xlabel("Kstar Angular Distance (rad)")
    plt.ylabel("Density")
    plt.title("Angular Distance  Distribution")
    plt.legend()
    plt.show()
    #%%
    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['momentum_diff'], bins=500 , alpha=0.7,density=True, label='background')
    plt.hist(sigdata['momentum_diff'], bins=500, alpha=0.7,density=True,label='signal')

    plt.xlabel("Missing Mass Squared (MeV^2)")
    plt.ylabel("Density")
    plt.title("Missing Mass Squared Distribution")
    plt.legend()
    plt.show()
    #%%
    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['theta_rad'], bins=500 , alpha=0.7,density=True, label='background')
    plt.hist(sigdata['theta_rad'], bins=500, alpha=0.7,density=True,label='signal')

    plt.xlabel("Angle between kaon and pion (rad)")
    plt.ylabel("Density")
    plt.title("Angle Distribution")
    plt.legend()
    plt.show()
# %%
