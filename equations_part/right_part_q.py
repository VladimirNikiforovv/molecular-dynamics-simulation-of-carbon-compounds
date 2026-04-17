import numpy as np
import numba as nb

@nb.njit
def dHdp(vec_P):
    m = 1243.7124
    # m = 12.0107 
    q_i_dot = vec_P/m
    
    return q_i_dot