import numpy as np
import numba as nb

@nb.njit
def f_cut_off(r):
    R_1 = 1.7
    R_2 = 2.0
    if r <= R_1:
        return 1.0
    elif r >= R_2:
        return 0.0
    else:
        return 0.5*(1.0 + np.cos((r - R_1) / (R_2 - R_1) * np.pi))
 
# @nb.njit(parallel=True)
# def B_ij_s(vec_q, i, j, N):
    
#     c_0 = 19.0
#     a_0 = 0.011304
#     d_0 = 2.5
    
#     delta = 0.80469
    
#     B_i_j = np.zeros((N), dtype=np.float64)
    
#     B_j_i = np.zeros((N), dtype=np.float64)
    
#     for k in nb.prange(N): 
#         if k != i and k != j:
#             abs_r_ik = np.sqrt(np.sum((vec_q[:,i]-vec_q[:,k])**2))
#             abs_r_jk = np.sqrt(np.sum((vec_q[:,j]-vec_q[:,k])**2))
#             abs_r_ij = np.sqrt(np.sum((vec_q[:,i]-vec_q[:,j])**2))
#             abs_r_ji = np.sqrt(np.sum((vec_q[:,j]-vec_q[:,i])**2))
            
#             G_i_j = a_0*(1 + (c_0/d_0)**2 - c_0**2/(d_0**2 + (1 + (np.sum((vec_q[:,i]-vec_q[:,j])*(vec_q[:,i]-vec_q[:,k]))/(abs_r_ij*abs_r_ik)))**2))
#             G_j_i = a_0*(1 + (c_0/d_0)**2 - c_0**2/(d_0**2 + (1 + (np.sum((vec_q[:,j]-vec_q[:,i])*(vec_q[:,j]-vec_q[:,k]))/(abs_r_ji*abs_r_jk)))**2))
            
#             B_i_j[k] = G_i_j * f_cut_off(abs_r_ik)
#             B_j_i[k] = G_j_i * f_cut_off(abs_r_jk)
    
    
#     bij = (1 + np.sum(B_i_j))**(-delta)
#     bji = (1 + np.sum(B_j_i))**(-delta)
#     B_s = (bij + bji)/2
    
#     return B_s 

@nb.njit(parallel=True)
def U(q_vec, n, N):
    D_e = 6.325
    S = 1.29
    beta = 1.5
    R_e = 1.315

    sum_u = np.zeros((N), dtype=np.float64)
    
    c_0 = 19.0
    a_0 = 0.011304
    d_0 = 2.5
    
    delta = 0.80469
    
    for j in nb.prange(N):
        if j!=n:
            r_nj = q_vec[:,n]-q_vec[:,j]
            r_nj_norm = np.sqrt(np.sum(r_nj**2))
            f_cut = f_cut_off(r_nj_norm)
            VR_nj = f_cut*(D_e/(S-1))*np.exp(-beta*np.sqrt(2*S)*(r_nj_norm-R_e))
            VA_nj = f_cut*(D_e*S/(S-1))*np.exp(-beta*np.sqrt(2/S)*(r_nj_norm-R_e))
            B_i_j = np.zeros((N), dtype=np.float64)
            B_j_i = np.zeros((N), dtype=np.float64)
            for k in nb.prange(N): 
                if k != n and k != j:
                    abs_r_ik = np.sqrt(np.sum((q_vec[:,n]-q_vec[:,k])**2))
                    abs_r_jk = np.sqrt(np.sum((q_vec[:,j]-q_vec[:,k])**2))
                    abs_r_ij = np.sqrt(np.sum((q_vec[:,n]-q_vec[:,j])**2))
                    abs_r_ji = np.sqrt(np.sum((q_vec[:,j]-q_vec[:,n])**2))
                    
                    G_i_j = a_0*(1 + (c_0/d_0)**2 - c_0**2/(d_0**2 + (1 + (np.sum((q_vec[:,n]-q_vec[:,j])*(q_vec[:,n]-q_vec[:,k]))/(abs_r_ij*abs_r_ik)))**2))
                    G_j_i = a_0*(1 + (c_0/d_0)**2 - c_0**2/(d_0**2 + (1 + (np.sum((q_vec[:,j]-q_vec[:,n])*(q_vec[:,j]-q_vec[:,k]))/(abs_r_ji*abs_r_jk)))**2))
                    
                    B_i_j[k] = G_i_j * f_cut_off(abs_r_ik)
                    B_j_i[k] = G_j_i * f_cut_off(abs_r_jk) 
            bij = (1 + np.sum(B_i_j))**(-delta)
            bji = (1 + np.sum(B_j_i))**(-delta)
            B_s = (bij + bji)/2      
            sum_u[j] = VR_nj - B_s * VA_nj
            
    U_n_parc = np.sum(sum_u)/2
    
    return U_n_parc

@nb.njit(parallel=True)
def dHdq(q_vec, n, N):

    # h_q = 1e-8
    h_q = 1e-3
    
    e_0h = np.array([1.0, 0.0, 0.0], dtype=np.float64)*h_q
    e_1h = np.array([0.0, 1.0, 0.0], dtype=np.float64)*h_q
    e_2h = np.array([0.0, 0.0, 1.0], dtype=np.float64)*h_q
    
    h_q = h_q*2
    
    q_cnt_p = np.copy(q_vec)
    q_cnt_p[:,n] = q_vec[:,n] + e_0h
    
    q_cnt_m = np.copy(q_vec)
    q_cnt_m[:,n] = q_vec[:,n] - e_0h
    
    dU_dx_0 = (U(q_cnt_p, n, N) - U(q_cnt_m, n, N))/h_q
    
    q_cnt_p = np.copy(q_vec)
    q_cnt_p[:,n] = q_vec[:,n] + e_1h
    
    q_cnt_m = np.copy(q_vec)
    q_cnt_m[:,n] = q_vec[:,n] - e_1h
    
    dU_dx_1 = (U(q_cnt_p, n, N) - U(q_cnt_m, n, N))/h_q
    
    q_cnt_p = np.copy(q_vec)
    q_cnt_p[:,n] = q_vec[:,n] + e_2h
    
    q_cnt_m = np.copy(q_vec)
    q_cnt_m[:,n] = q_vec[:,n] - e_2h
    
    dU_dx_2 = (U(q_cnt_p, n, N) - U(q_cnt_m, n, N))/h_q
    
    grad_u = -np.array([dU_dx_0,
                        dU_dx_1,
                        dU_dx_2], dtype=np.float64)
     
    return grad_u