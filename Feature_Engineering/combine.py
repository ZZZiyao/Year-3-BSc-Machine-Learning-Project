import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

def compute_combine(df):

    df['mT_mu_max'] = df[['mT_mu_minus', 'mT_mu_plus']].max(axis=1)
    df['DeltaPhipk_max'] = df[['DeltaPhip', 'DeltaPhik']].max(axis=1)
    df['fd_max']=df[['tau_fd_m','tau_fd_p']].max(axis=1)
    df['kstar_P_B0']=df['Kaon_P_B0']+df['Pion_P_B0']
    df['kstar_PT_B0']=df['Kaon_PT_B0']+df['Pion_PT_B0']
    df['jpsi_P_B0']=df['MP_P_B0']+df['MM_P_B0']
    df['jpsi_PT_B0']=df['MP_PT_B0']+df['MM_PT_B0']
    df['mu_P_B0_max'] = df[['MM_P_B0', 'MP_P_B0']].max(axis=1)
    df['mu_PT_B0_max'] = df[['MM_PT_B0', 'MP_PT_B0']].max(axis=1)
    return df







