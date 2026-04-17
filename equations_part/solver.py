import numpy as np
import numba as nb

from equations_part.right_part_q import dHdp
# from equations_part.right_part_P import dHdq
# from equations_part.right_part_p_opt import dHdq

from equations_part.numerical_grad_U import dHdq

@nb.njit
def H_sys(q_i, p_i, i, N):
    
    solv_vec = np.zeros((6), dtype=np.float64)
    
    solv_vec[:3] = dHdp(p_i)
    
    solv_vec[3:] = dHdq(q_i, i, N)
    
    return solv_vec


@nb.njit(parallel=True)
def runge_our(q_t, p_t, dt, N):
    q = np.copy(q_t) 
    p = np.copy(p_t) 
    
    k1_q = np.zeros((3, N))
    k1_p = np.zeros((3, N))
    k2_q = np.zeros((3, N))
    k2_p = np.zeros((3, N))
    k3_q = np.zeros((3, N))
    k3_p = np.zeros((3, N))
    k4_q = np.zeros((3, N))
    k4_p = np.zeros((3, N))
    
    # k1
    for i in nb.prange(N):
        k1_q[:, i] = dHdp(p[:, i])
        k1_p[:, i] = dHdq(q, i, N)
    
    # k2
    q_temp = q + k1_q * dt/2
    p_temp = p + k1_p * dt/2
    for i in nb.prange(N):
        k2_q[:, i] = dHdp(p_temp[:, i])
        k2_p[:, i] = dHdq(q_temp, i, N)
    
    # k3
    q_temp = q + k2_q * dt/2
    p_temp = p + k2_p * dt/2
    for i in nb.prange(N):
        k3_q[:, i] = dHdp(p_temp[:, i])
        k3_p[:, i] = dHdq(q_temp, i, N)
    
    # k4
    q_temp = q + k3_q * dt
    p_temp = p + k3_p * dt
    for i in nb.prange(N):
        k4_q[:, i] = dHdp(p_temp[:, i])
        k4_p[:, i] = dHdq(q_temp, i, N)
    
    # Final update
    q_new = q + dt * (k1_q + 2*k2_q + 2*k3_q + k4_q) / 6
    p_new = p + dt * (k1_p + 2*k2_p + 2*k3_p + k4_p) / 6
    
    return p_new, q_new


@nb.njit(parallel=True)
def symplectic_step(q, p, dt, N):
    """Базовый симплектический шаг (верлетовский тип)"""
    # Половина шага для импульсов
    for i in nb.prange(N):
        p[:, i] = p[:, i] + 0.5 * dt * dHdq(q, i, N)
    
    # Полный шаг для координат
    for i in nb.prange(N):
        q[:, i] = q[:, i] + dt * dHdp(p[:, i])
    
    # Вторая половина шага для импульсов
    for i in nb.prange(N):
        p[:, i] = p[:, i] + 0.5 * dt * dHdq(q, i, N)
    
    return q, p

@nb.njit
def yoshida_four(q_t, p_t, dt, N):
    
    q = np.copy(q_t) 
    p = np.copy(p_t) 
    
    
    """Интегрирует систему методом Йосиды 4-го порядка"""
    # Коэффициенты для метода Йосиды 4-го порядка
    w0 = -2**(1/3)/(2 - 2**(1/3))
    w1 = 1/(2 - 2**(1/3))
    c1 = c4 = w1/2
    c2 = c3 = (w0 + w1)/2
    d1 = d3 = w1
    d2 = w0
    
    # Первый шаг
    q_temp, p_temp = symplectic_step(q, p, c1*dt, N)
    
    # Второй шаг
    q_temp, p_temp = symplectic_step(q_temp, p_temp, d1*dt, N)
    
    # Третий шаг
    q_temp, p_temp = symplectic_step(q_temp, p_temp, c2*dt, N)
    
    # Четвертый шаг
    q_temp, p_temp = symplectic_step(q_temp, p_temp, d2*dt, N)
    
    # Пятый шаг
    q_temp, p_temp = symplectic_step(q_temp, p_temp, c3*dt, N)
    
    # Шестой шаг
    q_temp, p_temp = symplectic_step(q_temp, p_temp, d3*dt, N)
    
    # Седьмой шаг
    q_new, p_new = symplectic_step(q_temp, p_temp, c4*dt, N)
    
    return p_new, q_new