import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compute_tau_fly_distance_square(df):


    ref_mu_plus = df[['mu_plus_REFPX', 'mu_plus_REFPY', 'mu_plus_REFPZ']].values
    mom_mu_plus = df[['mu_plus_PX', 'mu_plus_PY', 'mu_plus_PZ']].values
    ref_mu_minus = df[['mu_minus_REFPX', 'mu_minus_REFPY', 'mu_minus_REFPZ']].values
    mom_mu_minus = df[['mu_minus_PX', 'mu_minus_PY', 'mu_minus_PZ']].values
    b0_end = df[['B0_ENDVERTEX_X', 'B0_ENDVERTEX_Y', 'B0_ENDVERTEX_Z']].values
    b0_own = df[['B0_OWNPV_X', 'B0_OWNPV_Y', 'B0_OWNPV_Z']].values

    # use x and z to calculate lambdas

    A = np.stack([b0_end[:, [0, 2]] - b0_own[:, [0, 2]], -mom_mu_plus[:, [0, 2]]], axis=2)
    B = (ref_mu_plus[:, [0, 2]] - b0_own[:, [0, 2]]).reshape(-1, 2, 1)
    C = np.stack([b0_end[:, [0, 2]] - b0_own[:, [0, 2]], -mom_mu_minus[:, [0, 2]]], axis=2)
    D = (ref_mu_minus[:, [0, 2]] - b0_own[:, [0, 2]]).reshape(-1, 2, 1)


    lambda_solutions_p = np.linalg.solve(A, B)
    lambda_solutions_m = np.linalg.solve(C, D)


    lambda1_p, lambda2_p = lambda_solutions_p[:, 0, 0], lambda_solutions_p[:, 1, 0]
    lambda1_m, lambda2_m = lambda_solutions_m[:, 0, 0], lambda_solutions_m[:, 1, 0]


    
    intersect_1_p = b0_own + lambda1_p[:, np.newaxis] * (b0_end - b0_own)
    intersect_2_p = ref_mu_plus + lambda2_p[:, np.newaxis] * mom_mu_plus
    intersect_1_m = b0_own + lambda1_m[:, np.newaxis] * (b0_end - b0_own)
    intersect_2_m = ref_mu_minus + lambda2_m[:, np.newaxis] * mom_mu_minus

    error_p_y = np.abs(intersect_1_p[:, 1] - intersect_2_p[:, 1])
    error_m_y = np.abs(intersect_1_m[:, 1] - intersect_2_m[:, 1])

    threshold = 1e-6  
    error_rows = np.where((error_p_y > threshold) | (error_m_y > threshold))[0]

    if len(error_rows) > 0:
        print(f"found {len(error_rows)} rows with error in y direction")
        print(df.iloc[error_rows[:10]])
    
    tau_fd_p = np.linalg.norm(intersect_1_p - b0_end, axis=1)  
    tau_fd_m = np.linalg.norm(intersect_1_m - b0_end, axis=1)

    
    df = df.copy()
    df[['tau_fd_p', 'tau_fd_m']] = np.column_stack([tau_fd_p, tau_fd_m])
    df[['inter_pX', 'inter_pY', 'inter_pZ']] = intersect_1_p
    df[['inter_mX', 'inter_mY', 'inter_mZ']] = intersect_1_m
    df[['error_p_y', 'error_m_y']] = np.column_stack([error_p_y, error_m_y])

    return df



def compute_tau_fly_distance(df):
    ref_mu_plus = df[['mu_plus_REFPX', 'mu_plus_REFPY', 'mu_plus_REFPZ']].values
    mom_mu_plus = df[['mu_plus_PX', 'mu_plus_PY', 'mu_plus_PZ']].values
    ref_mu_minus = df[['mu_minus_REFPX', 'mu_minus_REFPY', 'mu_minus_REFPZ']].values
    mom_mu_minus = df[['mu_minus_PX', 'mu_minus_PY', 'mu_minus_PZ']].values
    b0_end = df[['B0_ENDVERTEX_X', 'B0_ENDVERTEX_Y', 'B0_ENDVERTEX_Z']].values
    b0_own = df[['B0_OWNPV_X', 'B0_OWNPV_Y', 'B0_OWNPV_Z']].values

    N = len(df)  # 事件数
    lambda_solutions_p = np.zeros((N, 2))  # 预分配 (N,2)
    lambda_solutions_m = np.zeros((N, 2))  # 预分配 (N,2)

    for i in range(N):
        # 取出单个事件的数据
        A_p = np.column_stack([(b0_end[i] - b0_own[i]), -mom_mu_plus[i]])  # (3,2)
        B_p = (ref_mu_plus[i] - b0_own[i])  # (3,)

        A_m = np.column_stack([(b0_end[i] - b0_own[i]), -mom_mu_minus[i]])
        B_m = (ref_mu_minus[i] - b0_own[i])

        # 逐个事件求解
        lambda_solutions_p[i], _, _, _ = np.linalg.lstsq(A_p, B_p, rcond=None)
        lambda_solutions_m[i], _, _, _ = np.linalg.lstsq(A_m, B_m, rcond=None)

    # 提取 λ1 和 λ2
    lambda1_p, lambda2_p = lambda_solutions_p[:, 0], lambda_solutions_p[:, 1]
    lambda1_m, lambda2_m = lambda_solutions_m[:, 0], lambda_solutions_m[:, 1]

    # 计算交点
    intersect_1_p = b0_own + lambda1_p[:, np.newaxis] * (b0_end - b0_own)
    intersect_2_p = ref_mu_plus + lambda2_p[:, np.newaxis] * mom_mu_plus
    intersect_1_m = b0_own + lambda1_m[:, np.newaxis] * (b0_end - b0_own)
    intersect_2_m = ref_mu_minus + lambda2_m[:, np.newaxis] * mom_mu_minus

    # 计算误差（欧几里得距离）
    error_p = np.linalg.norm(intersect_1_p - intersect_2_p, axis=1)
    error_m = np.linalg.norm(intersect_1_m - intersect_2_m, axis=1)

    threshold = 1
    error_rows = np.where((error_p > threshold) | (error_m > threshold))[0]

    if len(error_rows) > 0:
        print(f"found {len(error_rows)} rows with error in Euclidean distance")
        print(df.iloc[error_rows[:10]])

    # 计算 tau fly distance
    tau_fd_p = np.linalg.norm(intersect_1_p - b0_end, axis=1)  
    tau_fd_m = np.linalg.norm(intersect_1_m - b0_end, axis=1)

    # 存入 df
    
    df[['tau_fd_p', 'tau_fd_m']] = np.column_stack([tau_fd_p, tau_fd_m])
    df[['error_p', 'error_m']] = np.column_stack([error_p, error_m])

    return df



def compute_tau_fly_distance_piliang(df):
    ref_mu_plus = df[['mu_plus_REFPX', 'mu_plus_REFPY', 'mu_plus_REFPZ']].values
    mom_mu_plus = df[['mu_plus_PX', 'mu_plus_PY', 'mu_plus_PZ']].values
    ref_mu_minus = df[['mu_minus_REFPX', 'mu_minus_REFPY', 'mu_minus_REFPZ']].values
    mom_mu_minus = df[['mu_minus_PX', 'mu_minus_PY', 'mu_minus_PZ']].values
    b0_end = df[['B0_ENDVERTEX_X', 'B0_ENDVERTEX_Y', 'B0_ENDVERTEX_Z']].values
    b0_own = df[['B0_OWNPV_X', 'B0_OWNPV_Y', 'B0_OWNPV_Z']].values

    # 计算 A 和 B
    A_p = np.stack([b0_end - b0_own, -mom_mu_plus], axis=2)  # (N, 3, 2)
    B_p = (ref_mu_plus - b0_own)[:, :, np.newaxis]  # (N, 3, 1)

    A_m = np.stack([b0_end - b0_own, -mom_mu_minus], axis=2)
    B_m = (ref_mu_minus - b0_own)[:, :, np.newaxis]

    # 计算广义逆矩阵并求解 lambda1, lambda2
    lambda_solutions_p = np.einsum('nij,njk->nik', np.linalg.pinv(A_p), B_p).squeeze(-1)  # (N, 2)
    lambda_solutions_m = np.einsum('nij,njk->nik', np.linalg.pinv(A_m), B_m).squeeze(-1)

    lambda1_p, lambda2_p = lambda_solutions_p[:, 0], lambda_solutions_p[:, 1]
    lambda1_m, lambda2_m = lambda_solutions_m[:, 0], lambda_solutions_m[:, 1]

    # 计算交点
    intersect_1_p = b0_own + lambda1_p[:, np.newaxis] * (b0_end - b0_own)
    intersect_2_p = ref_mu_plus + lambda2_p[:, np.newaxis] * mom_mu_plus
    intersect_1_m = b0_own + lambda1_m[:, np.newaxis] * (b0_end - b0_own)
    intersect_2_m = ref_mu_minus + lambda2_m[:, np.newaxis] * mom_mu_minus

    # 计算误差（欧几里得距离）
    error_p = np.linalg.norm(intersect_1_p - intersect_2_p, axis=1)
    error_m = np.linalg.norm(intersect_1_m - intersect_2_m, axis=1)

    threshold = 1
    error_rows = np.where((error_p > threshold) | (error_m > threshold))[0]

    if len(error_rows) > 0:
        print(f"found {len(error_rows)} rows with error in Euclidean distance")
        print(df.iloc[error_rows[:10]])

    # 计算 tau fly distance
    tau_fd_p = np.linalg.norm(intersect_1_p - b0_end, axis=1)  
    tau_fd_m = np.linalg.norm(intersect_1_m - b0_end, axis=1)

    # 存入 df
    df = df.copy()
    df[['tau_fd_p', 'tau_fd_m']] = np.column_stack([tau_fd_p, tau_fd_m])
    df[['inter_pX', 'inter_pY', 'inter_pZ']] = intersect_1_p
    df[['inter_mX', 'inter_mY', 'inter_mZ']] = intersect_1_m
    df[['error_p', 'error_m']] = np.column_stack([error_p, error_m])

    return df


if __name__ == "__main__":

    bg1 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg1.csv')
    sig1 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig1.csv')


    bg_data = compute_tau_fly_distance(bg1)
    sig_data = compute_tau_fly_distance(sig1)


    def plot_tau_fly_distance(bg_data, sig_data, column, xlabel):
        plt.figure(figsize=(10, 6))
        #plt.xlim(-100,1000)
        plt.hist(bg_data[column], bins=1000, color='blue', label='Background', alpha=0.5, density=True, 
                range=(0, 100)
                )
        plt.hist(sig_data[column], bins=1000, color='red', label='Signal', alpha=0.5, density=True, 
                range=(0, 100)
                )

        plt.xlabel(xlabel)
        plt.ylabel('Density')
        plt.legend()
        plt.title(f'Distribution of {xlabel}')
        plt.show()

    plot_tau_fly_distance(bg_data, sig_data, 'tau_fd_p', 'Tau fly distance (mu plus)')
    plot_tau_fly_distance(bg_data, sig_data, 'tau_fd_m', 'Tau fly distance (mu minus)')


    plot_tau_fly_distance(bg_data, sig_data, 'error_p', 'error (mu plus)')
    plot_tau_fly_distance(bg_data, sig_data, 'error_m', 'error (mu minus)')
