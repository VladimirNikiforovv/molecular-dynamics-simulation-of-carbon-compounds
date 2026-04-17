import numpy as np
import numba as nb
# import matplotlib.pyplot as plt


@nb.njit
def f_cut_off(r):
    R_1 = 1.7#e-10
    R_2 = 2.0#e-10
    if r < R_1:
        return 1
    elif R_1 < r < R_2:
        return (1/2)*(1+np.cos((r-R_1)/(R_2-R_1)*np.pi))
    elif r > R_2:
        return 0

@nb.njit
def df_cut_off_dr(r):
    R_1 = 1.7#e-10
    R_2 = 2.0#e-10
    if r < R_1:
        return 0
    elif R_1 < r < R_2:
        return -(1/2)*(np.sin((r-R_1)/(R_2-R_1)*np.pi))*(1/(R_2-R_1))*np.pi  
    elif r > R_2:
        return 0
    
@nb.njit
def V_A(r):
    D_e = 6.325
    S = 1.29
    beta = 1.5
    R_e = 1.315
    return f_cut_off(r)*(D_e*S/(S-1))*np.exp(-beta*np.sqrt(2/S)*(r-R_e))

@nb.njit
def V_R(r):
    D_e = 6.325
    S = 1.29
    beta = 1.5
    R_e = 1.315
    return f_cut_off(r)*(D_e/(S-1))*np.exp(-beta*np.sqrt(2*S)*(r-R_e))
    
@nb.njit
def dV_R_dr(r):
    D_e = 6.325
    S = 1.29
    beta = 1.5
    R_e = 1.315
    
    rez = df_cut_off_dr(r)*(D_e/(S-1))*np.exp(-beta*np.sqrt(2*S)*(r-R_e))
    rez -= f_cut_off(r)*(D_e/(S-1))*np.exp(-beta*np.sqrt(2*S)*(r-R_e))*(beta*np.sqrt(2*S))
    
    return rez

@nb.njit
def dV_A_dr(r):
    D_e = 6.325
    S = 1.29
    beta = 1.5
    R_e = 1.315
    
    rez = df_cut_off_dr(r)*(D_e*S/(S-1))*np.exp(-beta*np.sqrt(2/S)*(r-R_e))
    rez -= f_cut_off(r)*(D_e*S/(S-1))*np.exp(-beta*np.sqrt(2/S)*(r-R_e))*(beta*np.sqrt(2/S))
    
    return rez

@nb.njit
def G_c(vec_r_ij, vec_r_ik, abs_r_ij, abs_r_ik):
    
    c_0 = 19.0
    a_0 = 0.011304
    d_0 = 2.5
    
    return a_0*(1 + (c_0/d_0)**2 - c_0**2/(d_0**2 + (1 + (np.sum(vec_r_ij*vec_r_ik)/(abs_r_ij*abs_r_ik)))**2))

@nb.njit(parallel=True)
def B_ij_s(vec_q, i, j, N):
    
    delta = 0.80469
    
    B_i_j = np.zeros((N), dtype=np.float64)
    
    B_j_i = np.zeros((N), dtype=np.float64)
    
    for k in nb.prange(N): 
        if k != i and k != j:
            abs_r_ik = np.sqrt(np.sum((vec_q[:,i]-vec_q[:,k])**2))
            abs_r_jk = np.sqrt(np.sum((vec_q[:,j]-vec_q[:,k])**2))
            abs_r_ij = np.sqrt(np.sum((vec_q[:,i]-vec_q[:,j])**2))
            abs_r_ji = np.sqrt(np.sum((vec_q[:,j]-vec_q[:,i])**2))
            B_i_j[k] = G_c(vec_q[:,i]-vec_q[:,j], vec_q[:,i]-vec_q[:,k], abs_r_ij, abs_r_ik) * f_cut_off(abs_r_ik)
            B_j_i[k] = G_c(vec_q[:,j]-vec_q[:,i], vec_q[:,j]-vec_q[:,k], abs_r_ji, abs_r_jk) * f_cut_off(abs_r_jk)
    
    
    bij = (1 + np.sum(B_i_j))**(-delta)
    bji = (1 + np.sum(B_j_i))**(-delta)
    B_s = (bij + bji)/2
    
    return B_s, bij, bji

@nb.njit
def dB_ij_s(vec_q, i, j, N, B_ij, B_ji):
    delta = 0.80469
    c_0 = 19.0
    a_0 = 0.011304
    d_0 = 2.5
    
    # dB_s_ij = -(delta*B_ij**(-delta-1))
    # dB_s_ji = -(delta*B_ji**(-delta-1))
    
    dB_s_ij = -(delta/B_ij)
    dB_s_ji = -(delta/B_ji)
    
    sum_1 = np.zeros((N), dtype=np.float64)
    sum_2 = np.zeros((N), dtype=np.float64)
    
    for k in nb.prange(N): 
        if k != i and k != j:
            abs_r_ik = np.sqrt(np.sum((vec_q[:,i]-vec_q[:,k])**2))
            
            abs_r_ij = np.sqrt(np.sum((vec_q[:,i]-vec_q[:,j])**2))
            
            vec_r_ij = vec_q[:,i]-vec_q[:,j]
            vec_r_ik = vec_q[:,i]-vec_q[:,k]
            
            vec_r_ji = vec_q[:,j]-vec_q[:,i]
            vec_r_jk = vec_q[:,j]-vec_q[:,k]
            abs_r_ji = np.sqrt(np.sum((vec_q[:,j]-vec_q[:,i])**2))
            abs_r_jk = np.sqrt(np.sum((vec_q[:,j]-vec_q[:,k])**2))
                        
            # dG_dr_ij = (np.sum(vec_r_ij*vec_r_ik) / ((abs_r_ij**2)*abs_r_ik)) * (a_0*c_0**2) / (d_0**2 + (1 + (np.sum(vec_r_ij*vec_r_ik)/(abs_r_ij*abs_r_ik)))**2)**2
            
            # dG_dr_ji = (np.sum(vec_r_ji*vec_r_jk)/((abs_r_ji**2)*abs_r_jk)) * (a_0*c_0**2) / (d_0**2 + (1 + (np.sum(vec_r_ji*vec_r_jk)/(abs_r_ji*abs_r_jk)))**2)**2
            cnt1 = (1 + (np.sum(vec_r_ij*vec_r_ik)/(abs_r_ij*abs_r_ik)))
            dG_dr_ij =  (a_0*c_0**2) / ((d_0**2 + cnt1**2)**2)
            dG_dr_ij *= cnt1*2
            dG_dr_ij *= np.sum(vec_r_ij*vec_r_ik)/((abs_r_ij**2)*abs_r_ik)
            
            cnt2 = (1 + (np.sum(vec_r_ji*vec_r_jk)/(abs_r_ji*abs_r_jk)))
            dG_dr_ji = (a_0*c_0**2) / ((d_0**2 + cnt2**2)**2)
            dG_dr_ji *= cnt2*2
            dG_dr_ij *= np.sum(vec_r_ji*vec_r_jk)/((abs_r_ji**2)*abs_r_jk)
            
            sum_1[k] = f_cut_off(abs_r_ik) * dG_dr_ij
            sum_2[k] = f_cut_off(abs_r_jk) * dG_dr_ji
            
    return (dB_s_ij*np.sum(sum_1) + dB_s_ji*np.sum(sum_2))/2

@nb.njit
def dUdr(r, vec_q, i, j, N):
    
    Bs, B_ij, B_ji = B_ij_s(vec_q, i, j, N)
    dB_dr = dB_ij_s(vec_q, i, j, N, B_ij, B_ji)
    
    dV_Rdr = dV_R_dr(r) - Bs*dV_A_dr(r) - dB_dr*V_A(r)
    
    # dV_Rdr = dV_R_dr(r) - Bs* dV_A_dr(r)
    
    return dV_Rdr

@nb.njit(parallel=True)
def dHdq(vec_q, i, N):
    
    p_i_dot = np.zeros((3), dtype=np.float64)
    
    for j in nb.prange(N):
        if j != i:
            
            direction = vec_q[:,i]-vec_q[:,j]
            
            r_ij = np.sqrt(np.sum(direction**2))
            
            p_i_dot += -dUdr(r_ij, vec_q, i, j, N)*direction/r_ij
            
    
    return p_i_dot
    
# N = 100
# r = np.linspace(0,5, N)
# ra = np.zeros((N), dtype=np.float64)

# for i in range(N):

#     ra[i] = -dUdr(r[i])
    
# plt.plot(r, ra)
# plt.show()

