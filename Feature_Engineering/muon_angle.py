#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_angles(df):
    # Compute phi (azimuthal angle)
    phi1 = np.arctan2(df['mu_plus_PY'], df['mu_plus_PX'])
    phi2 = np.arctan2(df['mu_minus_PY'], df['mu_minus_PX'])
    dphi = np.abs(phi1 - phi2)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)  # Normalize Δφ
    df['mdphi']=dphi

    
    eta1 = df['mu_plus_ETA']
    eta2 = df['mu_minus_ETA']
    deta=np.abs(eta1 - eta2)
    df['mdeta'] = deta

    # Compute ΔR (angular distance)
    df['mdR'] = np.sqrt(deta**2 + dphi**2)
    
    # Compute momentum difference
    p1 = np.sqrt(df['mu_plus_PX']**2 + df['mu_plus_PY']**2 + df['mu_plus_PZ']**2)
    p2 = np.sqrt(df['mu_minus_PX']**2 + df['mu_minus_PY']**2 + df['mu_minus_PZ']**2)
    df['mmomentum_diff'] = np.abs(p1 - p2) / (p1 + p2)

    # Compute angle between trajectories
    dot_product = (df['mu_plus_PX'] * df['mu_minus_PX'] + 
                   df['mu_plus_PY'] * df['mu_minus_PY'] + 
                   df['mu_plus_PZ'] * df['mu_minus_PZ'])
    cos_theta = dot_product / (p1 * p2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Avoid numerical errors
    df['mtheta_rad'] = np.arccos(cos_theta)

    return df


if __name__ == "__main__":


    bg1 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_bg1.csv')

    sig1=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_sig1.csv')


    bgdata = compute_angles(bg1)
    sigdata = compute_angles(sig1)

#%%

    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['dphi'], bins=500 , alpha=0.7,density=True, label='background')
    plt.hist(sigdata['dphi'], bins=500, alpha=0.7,density=True,label='signal')

    plt.xlabel("Missing Mass (MeV)")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Missing Mass Distribution")
    plt.show()
#%%
    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['deta'], bins=500 , alpha=0.7,density=True, label='background')
    plt.hist(sigdata['deta'], bins=500, alpha=0.7,density=True,label='signal')

    plt.xlabel("Missing Mass Squared (MeV^2)")
    plt.ylabel("Density")
    plt.title("Missing Mass Squared Distribution")
    plt.legend()
    plt.show()
    #%%
    plt.figure(figsize=(10, 5))
    plt.hist(bgdata['dR'], bins=500 , alpha=0.7,density=True, label='background')
    plt.hist(sigdata['dR'], bins=500, alpha=0.7,density=True,label='signal')

    plt.xlabel("Missing Mass Squared (MeV^2)")
    plt.ylabel("Density")
    plt.title("Missing Mass Squared Distribution")
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

    plt.xlabel("Angle between muons (rad)")
    plt.ylabel("Density")
    plt.title("Abgle Distribution")
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

