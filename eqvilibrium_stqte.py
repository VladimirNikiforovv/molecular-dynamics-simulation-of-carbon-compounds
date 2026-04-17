import numpy as np
import numba as nb
import scipy as sp
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from equations_part.solver import runge_four, runge_our, yoshida_four
from equations_part.criate_grafen import create_graphene_lattice
from equations_part.right_part_P import V_A, V_R, B_ij_s
from equations_part.right_part_p_opt import dHdq


@nb.njit(parallel=True)
def calc_main(q_not, N, T):
    h = 0.1

    q_solv = np.zeros((3, N, T), dtype=np.float64)
    p_solv = np.zeros((3, N, T), dtype=np.float64)
    
    q_solv[:,:,0] = q_not
    
    for t in range(1,T):
        p_solv[:,:,t], q_solv[:,:,t] = runge_our(q_solv[:,:,t-1], p_solv[:,:,t-1], h, N)
        # p_solv[:,:,t], q_solv[:,:,t] = yoshida_four(q_solv[:,:,t-1], p_solv[:,:,t-1], h, N)

    return q_solv, p_solv

@nb.njit(parallel=True)
def energy(param):
    T = 500
    N = 4
    m = 1243.7124
    # q_0 = np.array([[0.000, param[0], param[2] ],
    #                 [0.000, param[1], param[3] ],
    #                 [0.000, 0.000, 0.000 ]], dtype=np.float64)
    q_0 = np.array([[0.000, param[0], param[2], param[4] ],
                    [0.000, param[1], param[3], param[5] ],
                    [0.000, 0.000, 0.000 , 0.000]], dtype=np.float64)
    
    q, p = calc_main(q_0, N, T)
    E_u = np.zeros((T), dtype=np.float64)
    E_k = np.zeros((T), dtype=np.float64)
    for t in range(T):
        U_t = 0.0
        for i in range(N):
            E_k[t] = (1/(2*m))*np.sum(p[:,i,t]**2)
            for j in range(N):  
                if j != i: 
                    r_ij = np.sqrt(np.sum((q[:,i,t] - q[:,j,t])**2))
                    if r_ij == 0:
                        U_t += 10000000000
                        r_ij = 1
                    B_ij_val = B_ij_s(q[:,:,t], i, j, N)[0]
    
                    pair_energy = V_R(r_ij) - B_ij_val * V_A(r_ij)
                    U_t += pair_energy
        E_u[t] = U_t/2
               
    # return np.abs(np.sum(E_u))
    return np.sum(E_k) + (np.sum(E_u))
    # return - (np.sum(E_u))

# q_0 = np.array([[0.000, 1.315, 0  ],
#                 [0.000, 0.0,1.315 ],
#                 [0.000, 0.000, 0.000 ]], dtype=np.float64)

# eu = energy(np.array([1.315, 0.0, 0.0, 1.315], dtype=np.float64))
# print(eu[1])
# plt.plot(eu[0])
# plt.show()


# initial_guess = [1.315, 0.0, 0.0, 1.315]
initial_guess = [0.9, 0.4, 0.4, -0.9, -0.0, -0.9]

# Настройки метода оптимизации
options = {
    'maxiter': 50,    # Максимальное количество итераций
    'disp': True        # Показывать процесс оптимизации
}
# bounds = [(0.1, 1.7), (0.1, 1.7), (0.1, 1.7), (0.1, 1.7)]
# Вызов оптимизатора
result = minimize(
    fun=energy,
    x0=initial_guess,
    method='L-BFGS-B',
    # bounds=bounds,
    options=options
)

print("\nРезультаты оптимизации:")
print(f"Успех: {result.success}")
print(f"Сообщение: {result.message}")
print(f"Значение функции в минимуме: {result.fun:.6f}")
print(f"Найденные параметры: {np.round(result.x, 6)}")
print(f"Количество итераций: {result.nit}")