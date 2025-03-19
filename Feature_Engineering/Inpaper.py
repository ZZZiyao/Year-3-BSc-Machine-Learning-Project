import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

import numpy as np

def compute_angle(p1, p2):

    dot_product = np.sum(p1 * p2, axis=1)  
    norm_p1 = np.linalg.norm(p1, axis=1)   
    norm_p2 = np.linalg.norm(p2, axis=1)   

    
    cos_theta = np.clip(dot_product / (norm_p1 * norm_p2), -1.0, 1.0)
    return np.arccos(cos_theta)  # 计算夹角（弧度）


def compute_invariant_mass(p4_1, p4_2):

    E_total = p4_1[:, 0] + p4_2[:, 0]
    
    px_total = p4_1[:, 1] + p4_2[:, 1]
    py_total = p4_1[:, 2] + p4_2[:, 2]
    pz_total = p4_1[:, 3] + p4_2[:, 3]

    M2 = E_total**2 - (px_total**2 + py_total**2 + pz_total**2)
    
    return np.sqrt(np.maximum(M2, 0))

def compute_PT_PE_Ptot(four_momentum):

    E = four_momentum[:, 0]  
    px, py, pz = four_momentum[:, 1], four_momentum[:, 2], four_momentum[:, 3] 

    PT = np.sqrt(px**2 + py**2)  
    P_total = np.sqrt(px**2 + py**2 + pz**2)  

    return PT, E, P_total  

def lorentz_transform_to_CM(p4, P_tau_tau):
    """
    Transform a four-momentum from the lab frame to the tau+ tau- center-of-mass frame.
    
    Parameters:
    p4 (array-like): Four-momentum [E, px, py, pz] of the particle in the lab frame.
    P_tau_tau (array-like): Four-momentum [E, px, py, pz] of the tau+ tau- system in the lab frame.
    
    Returns:
    numpy.ndarray: Four-momentum [E', px', py', pz'] of the particle in the tau+ tau- CM frame.
    """
    # Convert inputs to numpy arrays (ensure float type for calculations)
    p4 = np.array(p4, dtype=float)
    P_tau_tau = np.array(P_tau_tau, dtype=float)
    
    # Total energy and momentum of the tau-tau system in lab frame
    E_total = P_tau_tau[0]
    p_total = P_tau_tau[1:]      # 3-momentum (px, py, pz)
    
    # Compute boost velocity vector: v = p_total / E_total
    v_boost = p_total / E_total
    v2 = np.dot(v_boost, v_boost)  # magnitude squared of v_boost
    
    # If the tau-tau system is already at rest (v = 0), no transformation is needed
    if v2 == 0.0:
        return p4.copy()
    
    # Compute Lorentz factor gamma
    gamma = 1.0 / np.sqrt(1.0 - v2)
    
    # Construct the 4x4 Lorentz transformation matrix for the boost
    Lambda = np.eye(4)                   # start with identity matrix
    Lambda[0, 0] = gamma                 # time-time component
    Lambda[0, 1:] = -gamma * v_boost     # time-space components
    Lambda[1:, 0] = -gamma * v_boost     # space-time components (row)
    # space-space components:
    Lambda[1:, 1:] += (gamma - 1.0) * np.outer(v_boost, v_boost) / v2
    
    # Apply the Lorentz transformation: p4_prime = Lambda * p4 (matrix-vector product)
    p4_prime = Lambda.dot(p4)
    
    return p4_prime

def batch_lorentz_transform(p4_array, tau4_array):
    transformed = []
    for i in range(len(p4_array)):
        p4 = p4_array[i]          
        tau_p4 = tau4_array[i]    
        boosted_p4 = lorentz_transform_to_CM(p4, tau_p4)
        transformed.append(boosted_p4)
    return np.array(transformed)

def compute_paper(df):

    Kaon_four = np.column_stack((df['kaon_PE'], df['kaon_PX'], df['kaon_PY'], df['kaon_PZ']))
    Pion_four = np.column_stack((df['pion_PE'], df['pion_PX'], df['pion_PY'], df['pion_PZ']))
    MP_four = np.column_stack((df['mu_plus_PE'], df['mu_plus_PX'], df['mu_plus_PY'], df['mu_plus_PZ']))
    MM_four = np.column_stack((df['mu_minus_PE'], df['mu_minus_PX'], df['mu_minus_PY'], df['mu_minus_PZ']))
    tau_four = np.column_stack((df['tau_PE'], df['tau_PX'], df['tau_PY'], df['tau_PZ']))
    B0_four=np.column_stack((df['B0_PE_C'], df['B0_PX_C'], df['B0_PY_C'], df['B0_PZ_C']))

    # 计算 Kaon/Pion 与 muon 夹角
    df['K_MP_angle'] = compute_angle(Kaon_four[:,1:], MP_four[:,1:])
    df['K_MM_angle'] = compute_angle(Kaon_four[:,1:], MM_four[:,1:])
    df['P_MP_angle'] = compute_angle(Pion_four[:,1:], MP_four[:,1:])
    df['P_MM_angle'] = compute_angle(Pion_four[:,1:], MM_four[:,1:])

    # 计算 Kaon/Pion 与相反电荷 muon 夹角
    df['K_mu_angle'] = np.where(
        df['kaon_TRUEID'].to_numpy() == 321, 
        compute_angle(Kaon_four[:,1:], MM_four[:,1:]),  
        compute_angle(Kaon_four[:,1:], MP_four[:,1:])   
    )
    df['P_mu_angle'] = np.where(
        df['pion_TRUEID'].to_numpy() == 211, 
        compute_angle(Pion_four[:,1:], MM_four[:,1:]),  
        compute_angle(Pion_four[:,1:], MP_four[:,1:])   
    )
    df['B0_MM_angle']=compute_angle(B0_four[:,1:], MM_four[:,1:])
    df['B0_MP_angle']=compute_angle(B0_four[:,1:], MP_four[:,1:])

    df['K_lepton_mass'] = np.where(
    df['kaon_TRUEID'] == 321,  
    compute_invariant_mass(Kaon_four, MM_four),
    compute_invariant_mass(Kaon_four, MP_four)
)


    # Boost 到 tau 质量中心系
    Kaon_boosted_tau = batch_lorentz_transform(Kaon_four, tau_four)
    Pion_boosted_tau = batch_lorentz_transform(Pion_four, tau_four)
    MP_boosted_tau = batch_lorentz_transform(MP_four, tau_four)
    MM_boosted_tau = batch_lorentz_transform(MM_four, tau_four)
    B0_boosted_tau=batch_lorentz_transform(B0_four,tau_four)

    df[['Kaon_PX_tau', 'Kaon_PY_tau', 'Kaon_PZ_tau']] = Kaon_boosted_tau[:, 1:] 
    df[['Pion_PX_tau', 'Pion_PY_tau', 'Pion_PZ_tau']] = Pion_boosted_tau[:, 1:]
    df[['MP_PX_tau', 'MP_PY_tau', 'MP_PZ_tau']] = MP_boosted_tau[:, 1:]
    df[['MM_PX_tau', 'MM_PY_tau', 'MM_PZ_tau']] = MM_boosted_tau[:, 1:]

    # 计算 Kaon/Pion 和 muon 在 tau rest frame 下的角度
    df['K_MP_angle_tauframe'] = compute_angle(Kaon_boosted_tau[:,1:], MP_boosted_tau[:,1:])
    df['K_MM_angle_tauframe'] = compute_angle(Kaon_boosted_tau[:,1:], MM_boosted_tau[:,1:])
    df['P_MP_angle_tauframe'] = compute_angle(Pion_boosted_tau[:,1:], MP_boosted_tau[:,1:])
    df['P_MM_angle_tauframe'] = compute_angle(Pion_boosted_tau[:,1:], MM_boosted_tau[:,1:])

    # 计算 PT, PE, P_total
    df[['Kaon_PT_tau', 'Kaon_PE_tau', 'Kaon_P_tau']] = np.column_stack(compute_PT_PE_Ptot(Kaon_boosted_tau))
    df[['Pion_PT_tau', 'Pion_PE_tau', 'Pion_P_tau']] = np.column_stack(compute_PT_PE_Ptot(Pion_boosted_tau))
    df[['MP_PT_tau', 'MP_PE_tau', 'MP_P_tau']] = np.column_stack(compute_PT_PE_Ptot(MP_boosted_tau))
    df[['MM_PT_tau', 'MM_PE_tau', 'MM_P_tau']] = np.column_stack(compute_PT_PE_Ptot(MM_boosted_tau))

    df['K_mu_angle_tauframe'] = np.where(
        df['kaon_TRUEID'].to_numpy() == 321, 
        compute_angle(Kaon_boosted_tau[:,1:], MM_boosted_tau[:,1:]),  
        compute_angle(Kaon_boosted_tau[:,1:], MP_boosted_tau[:,1:])   
    )
    df['B0_MM_angle_tauframe']=compute_angle(B0_boosted_tau[:,1:], MM_boosted_tau[:,1:])
    df['B0_MP_angle_tauframe']=compute_angle(B0_boosted_tau[:,1:], MP_boosted_tau[:,1:])

    df['muon_angle_tauframe'] = compute_angle(MM_boosted_tau[:,1:], MP_boosted_tau[:,1:])
    df['kp_angle_tauframe'] = compute_angle(Kaon_boosted_tau[:,1:], Pion_boosted_tau[:,1:])

    df['K_lepton_mass_tau'] = np.where(
    df['kaon_TRUEID'] == 321,  
    compute_invariant_mass(Kaon_boosted_tau, MM_boosted_tau),
    compute_invariant_mass(Kaon_boosted_tau, MP_boosted_tau)
)


    # Boost 到 B0 质量中心系
    Kaon_boosted_B0 = batch_lorentz_transform(Kaon_four, B0_four)
    Pion_boosted_B0 = batch_lorentz_transform(Pion_four, B0_four)
    MP_boosted_B0 = batch_lorentz_transform(MP_four, B0_four)
    MM_boosted_B0 = batch_lorentz_transform(MM_four, B0_four)
    B0_boosted_B0 = batch_lorentz_transform(B0_four, B0_four)


    df[['Kaon_PX_B0', 'Kaon_PY_B0', 'Kaon_PZ_B0']] = Kaon_boosted_B0[:, 1:] 
    df[['Pion_PX_B0', 'Pion_PY_B0', 'Pion_PZ_B0']] = Pion_boosted_B0[:, 1:]
    df[['MP_PX_B0', 'MP_PY_B0', 'MP_PZ_B0']] = MP_boosted_B0[:, 1:]
    df[['MM_PX_B0', 'MM_PY_B0', 'MM_PZ_B0']] = MM_boosted_B0[:, 1:]

    # 计算 Kaon/Pion 和 muon 在 B0 rest frame 下的角度
    df['K_MP_angle_B0frame'] = compute_angle(Kaon_boosted_B0[:,1:], MP_boosted_B0[:,1:])
    df['K_MM_angle_B0frame'] = compute_angle(Kaon_boosted_B0[:,1:], MM_boosted_B0[:,1:])
    df['P_MP_angle_B0frame'] = compute_angle(Pion_boosted_B0[:,1:], MP_boosted_B0[:,1:])
    df['P_MM_angle_B0frame'] = compute_angle(Pion_boosted_B0[:,1:], MM_boosted_B0[:,1:])

    # 计算 PT, PE, P_total
    df[['Kaon_PT_B0', 'Kaon_PE_B0', 'Kaon_P_B0']] = np.column_stack(compute_PT_PE_Ptot(Kaon_boosted_B0))
    df[['Pion_PT_B0', 'Pion_PE_B0', 'Pion_P_B0']] = np.column_stack(compute_PT_PE_Ptot(Pion_boosted_B0))
    df[['MP_PT_B0', 'MP_PE_B0', 'MP_P_B0']] = np.column_stack(compute_PT_PE_Ptot(MP_boosted_B0))
    df[['MM_PT_B0', 'MM_PE_B0', 'MM_P_B0']] = np.column_stack(compute_PT_PE_Ptot(MM_boosted_B0))

    df['B0_MM_angle_B0frame']=compute_angle(B0_boosted_B0[:,1:], MM_boosted_B0[:,1:])
    df['B0_MP_angle_B0frame']=compute_angle(B0_boosted_B0[:,1:], MP_boosted_B0[:,1:])


    df['K_lepton_mass_B0'] = np.where(
    df['kaon_TRUEID'] == 321,  
    compute_invariant_mass(Kaon_boosted_B0, MM_boosted_B0),
    compute_invariant_mass(Kaon_boosted_B0, MP_boosted_B0)
)

    df['K_mu_angle_B0frame'] = np.where(
        df['kaon_TRUEID'].to_numpy() == 321, 
        compute_angle(Kaon_boosted_B0[:,1:], MM_boosted_B0[:,1:]),  
        compute_angle(Kaon_boosted_B0[:,1:], MP_boosted_B0[:,1:])   
    )
    df['muon_angle_B0frame'] = compute_angle(MM_boosted_B0[:,1:], MP_boosted_B0[:,1:])
    df['kp_angle_B0frame'] = compute_angle(Kaon_boosted_B0[:,1:], Pion_boosted_B0[:,1:])

    
    df['Missing_PE']=df['B0_PE_C']-df['kaon_PE']-df['pion_PE']-df['mu_plus_PE']-df['mu_minus_PE']
    df['Missing_P']=df['B0_P_C']-df['kaon_P']-df['pion_P']-df['mu_plus_P']-df['mu_minus_P']
    df['Missing_PT']=df['B0_PT_C']-df['kaon_PT']-df['pion_PT']-df['mu_plus_PT']-df['mu_minus_PT']

    return df