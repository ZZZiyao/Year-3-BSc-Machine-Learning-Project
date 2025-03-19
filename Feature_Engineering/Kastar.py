import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

def kstar(df):

    df['kdelz'] = df['kstar_ENDVERTEX_Z'] - df['B0_ENDVERTEX_Z']

    df['max_k_PT'] = df[['kaon_PT', 'pion_PT']].max(axis=1)
    df['min_k_PT'] = df[['kaon_PT', 'pion_PT']].min(axis=1)

    df['max_k_PE'] = df[['kaon_PE', 'pion_PE']].max(axis=1)
    df['min_k_PE'] = df[['kaon_PE', 'pion_PE']].min(axis=1)

    df['max_k_P'] = df[['kaon_P', 'pion_P']].max(axis=1)
    df['min_k_P'] = df[['kaon_P', 'pion_P']].min(axis=1)

    df['max_k_PZ'] = df[['kaon_PZ', 'pion_PZ']].max(axis=1)
    df['min_k_PZ'] = df[['kaon_PZ', 'pion_PZ']].min(axis=1)

    df['max_k_ETA'] = df[['kaon_ETA', 'pion_ETA']].max(axis=1)
    df['min_k_ETA'] = df[['kaon_ETA', 'pion_ETA']].min(axis=1)

    return df

