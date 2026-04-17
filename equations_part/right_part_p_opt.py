import numpy as np
import numba as nb
# import matplotlib.pyplot as plt


# @nb.njit
# def f_cut_off(r):
#     R_1 = 1.7#e-10
#     R_2 = 2.0#e-10
#     if r <= R_1:
#         return 1.0
#     elif R_1 < r < R_2:
#         return (1/2)*(1+np.cos((r-R_1)/(R_2-R_1)*np.pi))
#     elif r >= R_2:
#         return 0.0

# @nb.njit
# def df_cut_off_dr(r):
#     R_1 = 1.7#e-10
#     R_2 = 2.0#e-10
#     if r <= R_1:
#         return 0.0
#     elif R_1 < r < R_2:
#         return -(1/2)*(np.sin((r-R_1)/(R_2-R_1)*np.pi))*(1/(R_2-R_1))*np.pi  
#     elif r >= R_2:
#         return 0.0
@nb.njit
def clip_scalar(x, lo, hi):
    if x < lo:
        return lo
    elif x > hi:
        return hi
    else:
        return x
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

@nb.njit
def df_cut_off_dr(r):
    R_1 = 1.7
    R_2 = 2.0
    if r <= R_1 or r >= R_2:
        return 0.0
    else:
        return -0.5 * np.sin((r - R_1) / (R_2 - R_1) * np.pi) * (1.0 / (R_2 - R_1)) * np.pi

@nb.njit(parallel=True)
def dHdq(vec_q, n, N):
    
    p_i_dot_N = np.zeros((3, N), dtype=np.float64)
    p_i_dot = np.zeros((3), dtype=np.float64)
    
    p_i_dot_term_2 = np.zeros((3), dtype=np.float64)
    
    D_e = 6.325
    S = 1.29
    beta = 1.5
    R_e = 1.315
    
    delta = 0.80469
    
    c_0 = 19.0
    a_0 = 0.011304
    d_0 = 2.5
    
    # for j in range(N):
    for j in nb.prange(N):
        if j != n:
            
            r_nj = vec_q[:,n]-vec_q[:,j]
            
            abs_r_nj = np.sqrt(np.sum(r_nj**2))
            
            df_cut = df_cut_off_dr(abs_r_nj)
            
            f_cut = f_cut_off(abs_r_nj)
            
            exp_vr = np.exp(-beta*np.sqrt(2*S)*(abs_r_nj-R_e))
            
            dVR_dr = (df_cut*(D_e/(S-1))*exp_vr
                      - f_cut*(D_e/(S-1))*exp_vr*(beta*np.sqrt(2*S)) )
            
            exp_va = np.exp(-beta*np.sqrt(2/S)*(abs_r_nj-R_e))
            
            dVA_dr = (df_cut*(D_e*S/(S-1))*exp_va
                       - f_cut*(D_e*S/(S-1))*exp_va*(beta*np.sqrt(2/S)))
            
            symmetry_nj = 0
            for k in range(N):
                if k != n and k != j:
                    
                    r_nk = vec_q[:,n]-vec_q[:,k]
                    
                    abs_r_nk = np.sqrt(np.sum(r_nk**2))
                    
                    cos_thet_njk = np.sum(r_nj*r_nk)/(abs_r_nj*abs_r_nk)
                    cos_thet_njk = clip_scalar(cos_thet_njk, -1.0, 1.0)
                    G_thet_ijk = a_0*(1 + (c_0/d_0)**2 - c_0**2/(d_0**2 + (1 + (cos_thet_njk))**2))
                    
                    f_nk = f_cut_off(abs_r_nk)
                    
                    symmetry_nj += G_thet_ijk*f_nk
            
            B_nj = (1 + symmetry_nj)**(-delta)
            
            symmetry_jn = 0
            for k in range(N):
                if k != n and k != j:
                    
                    r_jk = vec_q[:,j]-vec_q[:,k]
                    
                    abs_r_jk = np.sqrt(np.sum(r_jk**2))
                    
                    # r_jn = r_nj
                    cos_thet_jnk = np.sum(-r_nj*r_jk)/(abs_r_nj*abs_r_jk)
                    cos_thet_jnk = clip_scalar(cos_thet_jnk, -1.0, 1.0)
                    # G_thet_jik = a_0*(1 + (c_0/d_0)**2 - c_0**2/(d_0**2 + (1 + (np.sum(r_ji*r_jk)/(abs_r_ji*abs_r_jk)))**2))
                    G_thet_jik = a_0*(1 + (c_0/d_0)**2 - c_0**2/(d_0**2 + (1 + (cos_thet_jnk))**2))
                    
                    f_jk = f_cut_off(abs_r_jk)
                    
                    symmetry_jn += G_thet_jik*f_jk
                    
            B_jn = (1 + symmetry_jn)**(-delta)
            
            B_nj_s = (B_nj + B_jn)/2
            
            p_i_dot_N[:,j] = -(dVR_dr/abs_r_nj)*r_nj + B_nj_s*(dVA_dr/abs_r_nj)*r_nj
            # p_i_dot += (-dVR_dr + B_nj_s*dVA_dr)*r_nj/abs_r_nj
    p_i_dot[0] = np.sum(p_i_dot_N[0,:])
    p_i_dot[1] = np.sum(p_i_dot_N[1,:])
    p_i_dot[2] = np.sum(p_i_dot_N[2,:])
    # sum2
    # symmetry_ijk
    term_2_part_ij = np.zeros((3), dtype=np.float64)
    term_2_part_ij_N = np.zeros((3,N,N), dtype=np.float64)
    for i in nb.prange(N):
        for j in range(N):
            if j != i:
                r_ij = vec_q[:,i]-vec_q[:,j]
                
                abs_r_ij = np.sqrt(np.sum(r_ij**2))
                
                f_cut_r_ij = f_cut_off(abs_r_ij)
                
                V_A_r_ij = f_cut_r_ij*(D_e*S/(S-1))*np.exp(-beta*np.sqrt(2/S)*(abs_r_ij-R_e))
                
                K_ijk = 1
                for kk in range(N):
                    if kk!= i and kk!=j:
                        r_ij = vec_q[:,i]-vec_q[:,j]
                        abs_r_ij = np.sqrt(np.sum(r_ij**2))
                        r_ik = vec_q[:,i]-vec_q[:,kk]
                        abs_r_ik = np.sqrt(np.sum(r_ik**2))
                        cos_ijk = np.sum(r_ij*r_ik)/(abs_r_ij*abs_r_ik)
                        cos_ijk = clip_scalar(cos_ijk, -1.0, 1.0)
                        G_thet_ijk = a_0*(1 + (c_0/d_0)**2 - c_0**2/(d_0**2 + (1 + (cos_ijk))**2))
                        f_cut_ijk = f_cut_off(abs_r_ik)
                        K_ijk += f_cut_ijk*G_thet_ijk
                        
                prefactor = -V_A_r_ij*delta*(K_ijk**(-delta-1))
                
                sum_k_ij = np.zeros((3), dtype=np.float64)
                for k in range(N):
                    if k!= i and k!=j:
                        # term_1
                        term1 = np.zeros((3), dtype=np.float64)
                        if i == n:
                            r_ij = vec_q[:,i]-vec_q[:,j]
                            abs_r_ij = np.sqrt(np.sum(r_ij**2))
                            r_ik = vec_q[:,i]-vec_q[:,k]
                            abs_r_ik = np.sqrt(np.sum(r_ik**2))
                            df_cut_ik = df_cut_off_dr(abs_r_ik)
                            f_cut_ik = f_cut_off(abs_r_ik)
                            cos_ijk = np.sum(r_ij*r_ik)/(abs_r_ij*abs_r_ik)
                            cos_ijk = clip_scalar(cos_ijk, -1.0, 1.0)
                            G_ijk = a_0*(1 + (c_0/d_0)**2 - c_0**2/(d_0**2 + (1 + (cos_ijk))**2))
                            H_ijk = 2*(a_0*c_0**2*(1+cos_ijk)) / ((d_0**2 + (1+cos_ijk)**2)**2)
                            term1 -= (r_ik+r_ij)/(abs_r_ij*abs_r_ik)
                            term1 += cos_ijk*r_ij/(abs_r_ij**2)
                            term1 += cos_ijk*r_ik/(abs_r_ik**2)
                            term1 *= f_cut_ik*H_ijk
                            term1 += G_ijk*df_cut_ik*r_ik/abs_r_ik
                            
                            
                        # term_2
                        term2 = np.zeros((3), dtype=np.float64)
                        if k == n:
                            r_ij = vec_q[:,i]-vec_q[:,j]
                            abs_r_ij = np.sqrt(np.sum(r_ij**2))
                            r_ik = vec_q[:,i]-vec_q[:,k]
                            abs_r_ik = np.sqrt(np.sum(r_ik**2))
                            f_cut_ik = f_cut_off(abs_r_ik)
                            df_cut_ik = df_cut_off_dr(abs_r_ik)
                            cos_ijk = np.sum(r_ij*r_ik)/(abs_r_ij*abs_r_ik)
                            cos_ijk = clip_scalar(cos_ijk, -1.0, 1.0)
                            H_ijk = 2*(a_0*c_0**2*(1+cos_ijk)) / ((d_0**2 + (1+cos_ijk)**2)**2)
                            G_ijk = a_0*(1 + (c_0/d_0)**2 - c_0**2/(d_0**2 + (1 + (cos_ijk))**2))
                            term2 -= f_cut_ik*H_ijk*r_ij/(abs_r_ij*abs_r_ik)
                            term2 += G_ijk*df_cut_ik*r_ik/abs_r_ik
                            term2 += (f_cut_ik*H_ijk*cos_ijk/(abs_r_ik**2))*r_ik
                            
                            
                            
                        # term_3
                        term3 = np.zeros((3), dtype=np.float64)
                        if j == n:
                            r_ij = vec_q[:,i]-vec_q[:,j]
                            abs_r_ij = np.sqrt(np.sum(r_ij**2))
                            r_ik = vec_q[:,i]-vec_q[:,k]
                            abs_r_ik = np.sqrt(np.sum(r_ik**2))
                            f_cut_ik = f_cut_off(abs_r_ik)
                            cos_ijk = np.sum(r_ij*r_ik)/(abs_r_ij*abs_r_ik)
                            cos_ijk = clip_scalar(cos_ijk, -1.0, 1.0)
                            H_ijk = 2*(a_0*c_0**2*(1+cos_ijk)) / ((d_0**2 + (1+cos_ijk)**2)**2)
                            term3 += f_cut_ik*H_ijk*((cos_ijk/(abs_r_ij**2))*r_ij - r_ik/(abs_r_ij*abs_r_ik))
                            
                        sum_k_ij += (term1 - term2 - term3)
                        
                term_2_part_ij_N[:,i,j] = prefactor*sum_k_ij/4
                # term_2_part_ij += prefactor*sum_k_ij/4
    term_2_part_ij[0] = np.sum(term_2_part_ij_N[0,:,:])
    term_2_part_ij[1] = np.sum(term_2_part_ij_N[1,:,:])
    term_2_part_ij[2] = np.sum(term_2_part_ij_N[2,:,:])
                
    term_2_part_ji = np.zeros((3), dtype=np.float64)
    term_2_part_ji_N = np.zeros((3,N,N), dtype=np.float64)
    for i in nb.prange(N):
        for j in range(N):
            if j != i:
                r_ji = vec_q[:,j]-vec_q[:,i]
                
                abs_r_ji = np.sqrt(np.sum(r_ji**2))
                
                f_cut_r_ji = f_cut_off(abs_r_ji)
                
                V_A_r_ji = f_cut_r_ji*(D_e*S/(S-1))*np.exp(-beta*np.sqrt(2/S)*(abs_r_ji-R_e))
                
                K_jik = 1
                for kk in range(N):
                    if kk!= j and kk!=i:
                        r_ji = vec_q[:,j]-vec_q[:,i]
                        abs_r_ji = np.sqrt(np.sum(r_ji**2))
                        r_jk = vec_q[:,j]-vec_q[:,kk]
                        abs_r_jk = np.sqrt(np.sum(r_jk**2))
                        cos_jik = np.sum(r_ji*r_jk)/(abs_r_ji*abs_r_jk)
                        cos_jik = clip_scalar(cos_jik, -1.0, 1.0)
                        G_thet_jik = a_0*(1 + (c_0/d_0)**2 - c_0**2/(d_0**2 + (1 + (cos_jik))**2))
                        f_cut_jk = f_cut_off(abs_r_jk)
                        K_jik += f_cut_jk*G_thet_jik
                        
                prefactor = -V_A_r_ji*delta*(K_jik**(-delta-1))
                
                sum_k_ji = np.zeros((3), dtype=np.float64)
                for k in range(N):
                    if k!= i and k!=j:
                        # term_1
                        term1 = np.zeros((3), dtype=np.float64)
                        if j == n:
                            r_ji = vec_q[:,j]-vec_q[:,i]
                            abs_r_ji = np.sqrt(np.sum(r_ji**2))
                            r_jk = vec_q[:,j]-vec_q[:,k]
                            abs_r_jk = np.sqrt(np.sum(r_jk**2))
                            df_cut_jk = df_cut_off_dr(abs_r_jk)
                            f_cut_jk = f_cut_off(abs_r_jk)
                            cos_jik = np.sum(r_ji*r_jk)/(abs_r_ji*abs_r_jk)
                            cos_jik = clip_scalar(cos_jik, -1.0, 1.0)
                            G_jik = a_0*(1 + (c_0/d_0)**2 - c_0**2/(d_0**2 + (1 + (cos_jik))**2))
                            H_jik = 2*(a_0*c_0**2*(1+cos_jik)) / ((d_0**2 + (1+cos_jik)**2)**2)
                            term1 -= (r_jk+r_ji)/(abs_r_ji*abs_r_jk)
                            term1 += cos_jik*r_ji/(abs_r_ji**2)
                            term1 += cos_jik*r_jk/(abs_r_jk**2)
                            term1 *= f_cut_jk*H_jik
                            term1 += G_jik*df_cut_jk*r_jk/abs_r_jk
                            
                            
                        # term_2
                        term2 = np.zeros((3), dtype=np.float64)
                        if k == n:
                            r_ji = vec_q[:,j]-vec_q[:,i]
                            abs_r_ji = np.sqrt(np.sum(r_ji**2))
                            r_jk = vec_q[:,j]-vec_q[:,k]
                            abs_r_jk = np.sqrt(np.sum(r_jk**2))
                            f_cut_jk = f_cut_off(abs_r_jk)
                            df_cut_jk = df_cut_off_dr(abs_r_jk)
                            cos_jik = np.sum(r_ji*r_jk)/(abs_r_ji*abs_r_jk)
                            cos_jik = clip_scalar(cos_jik, -1.0, 1.0)
                            H_jik = 2*(a_0*c_0**2*(1+cos_jik)) / ((d_0**2 + (1+cos_jik)**2)**2)
                            G_jik = a_0*(1 + (c_0/d_0)**2 - c_0**2/(d_0**2 + (1 + (cos_jik))**2))
                            term2 -= f_cut_jk*H_jik*r_ji/(abs_r_ji*abs_r_jk)
                            term2 += G_jik*df_cut_jk*r_jk/abs_r_jk
                            term2 += (f_cut_jk*H_jik*cos_jik/(abs_r_jk**2))*r_jk
                            
                            
                            
                        # # term_3
                        term3 = np.zeros((3), dtype=np.float64)
                        if i == n:
                            r_ji = vec_q[:,j]-vec_q[:,i]
                            abs_r_ji = np.sqrt(np.sum(r_ji**2))
                            r_jk = vec_q[:,j]-vec_q[:,k]
                            abs_r_jk = np.sqrt(np.sum(r_jk**2))
                            f_cut_jk = f_cut_off(abs_r_jk)
                            cos_jik = np.sum(r_ji*r_jk)/(abs_r_ji*abs_r_jk)
                            cos_jik = clip_scalar(cos_jik, -1.0, 1.0)
                            H_jik = 2*(a_0*c_0**2*(1+cos_jik)) / ((d_0**2 + (1+cos_jik)**2)**2)
                            term3 += f_cut_jk*H_jik*((cos_jik/(abs_r_ji**2))*r_ji - r_jk/(abs_r_ji*abs_r_jk))
                            
                        sum_k_ji += (term1 - term2 - term3)
                # term_2_part_ji += prefactor*sum_k_ji/4
                term_2_part_ji_N[:,i,j] = prefactor*sum_k_ji/4
    term_2_part_ji[0] = np.sum(term_2_part_ji_N[0,:,:])
    term_2_part_ji[1] = np.sum(term_2_part_ji_N[1,:,:])
    term_2_part_ji[2] = np.sum(term_2_part_ji_N[2,:,:])
                
    return p_i_dot + term_2_part_ij + term_2_part_ji