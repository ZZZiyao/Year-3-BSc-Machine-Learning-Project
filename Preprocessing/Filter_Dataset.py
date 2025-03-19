import pandas as pd

bg_files = [
    "D:/Year3/BSc Project/Particle-Machine-Learning/dataset/bd2ddkst_munu_MU_2016_1013_reducedbranches.pkl",
    "D:/Year3/BSc Project/Particle-Machine-Learning/dataset/bd2ddkst_munu_MD_2016_1014_reducedbranches.pkl",
    "D:/Year3/BSc Project/Particle-Machine-Learning/dataset/bd2ddkst_munu_MD_2017_1016_reducedbranches.pkl",
    "D:/Year3/BSc Project/Particle-Machine-Learning/dataset/bd2ddkst_munu_MD_2018_1018_reducedbranches.pkl",
    "D:/Year3/BSc Project/Particle-Machine-Learning/dataset/bd2ddkst_munu_MU_2017_1015_reducedbranches.pkl",
    "D:/Year3/BSc Project/Particle-Machine-Learning/dataset/bd2ddkst_munu_MU_2018_1017_reducedbranches.pkl"
]

sig_files = [
    "D:/Year3/BSc Project/Particle-Machine-Learning/dataset/output_kpitautau_2016_md_reducedbranches.pkl",
    "D:/Year3/BSc Project/Particle-Machine-Learning/dataset/output_kpitautau_2016_mu_reducedbranches.pkl",
    "D:/Year3/BSc Project/Particle-Machine-Learning/dataset/output_kpitautau_2017_md_reducedbranches.pkl",
    "D:/Year3/BSc Project/Particle-Machine-Learning/dataset/output_kpitautau_2017_mu_reducedbranches.pkl",
    "D:/Year3/BSc Project/Particle-Machine-Learning/dataset/output_kpitautau_2018_md_reducedbranches.pkl",
    "D:/Year3/BSc Project/Particle-Machine-Learning/dataset/output_kpitautau_2018_mu_reducedbranches.pkl"
]

mu_plus_ids = [-13]
mu_minus_ids = [13]
kaon_ids = [321, -321]
pion_ids = [211, -211]
kp_mother = [-511, 511, 313, -313]
gd=[-511,511]

bg_mu_p_mother = [411]
bg_mu_m_mother = [-411]
sig_mu_p_mother = [-15]
sig_mu_m_mother = [15]

def filter_and_save(file_list, prefix, mu_p_mother, mu_m_mother):

    for i, file_path in enumerate(file_list, start=1):
        print(f"Processing file: {file_path}")
        df = pd.read_pickle(file_path)

    
        filtered_df = df[
            (df['mu_plus_TRUEID'].isin(mu_plus_ids)) &
            (df['mu_minus_TRUEID'].isin(mu_minus_ids)) &
            (df['kaon_TRUEID'].isin(kaon_ids)) &
            (df['pion_TRUEID'].isin(pion_ids)) &
            (df['kaon_MC_MOTHER_ID'].isin(kp_mother)) &
            (df['pion_MC_MOTHER_ID'].isin(kp_mother)) &
            (df['mu_plus_MC_GD_MOTHER_ID'].isin(gd)) &
            (df['mu_minus_MC_GD_MOTHER_ID'].isin(gd)) &
            (df['mu_plus_MC_MOTHER_ID'].isin(mu_p_mother)) &
            (df['mu_minus_MC_MOTHER_ID'].isin(mu_m_mother))
        ]

        
        save_path = f"filtered_{prefix}{i}.csv"
        filtered_df.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")


filter_and_save(bg_files, "bg", bg_mu_p_mother, bg_mu_m_mother)
filter_and_save(sig_files, "sig", sig_mu_p_mother, sig_mu_m_mother)
