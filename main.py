import numpy as np
import numba as nb
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# from equations_part.solver import runge_four, runge_our, yoshida_four
from equations_part.solver import runge_our, yoshida_four
from equations_part.criate_grafen import create_graphene_lattice, create_carbon_ring
from equations_part.right_part_P import V_A, V_R, B_ij_s
from equations_part.right_part_p_opt import dHdq

# @nb.njit(parallel=True)
# def energy(p, q, N, T):
#     E_k = np.zeros((T), dtype=np.float64)
#     E_u = np.zeros((T), dtype=np.float64)
#     # m = 12.0107 * 1836.15267389
#     m = 12.0107 * 103.6427
#     for t in range(T):
#         for i in range(N):
#             E_k[t] += np.sum(p[:,i,t]*p[:,i,t])/(m)/2
#             for j in nb.prange(N):
#                 if j != i:
#                     r_ij = np.sqrt(np.sum((q[:,i,t] - q[:,j,t])**2))
#                     E_u[t] += (V_R(r_ij) - B_ij_s(q[:,:,t], i, j, N)[0]*V_A(r_ij))/2 #B_ij_s(q[:,:,t], i, j, N)[0]
#     return E_k, E_u

@nb.njit(parallel=True)
def energy(p, q, N, T):
    m = 1243.7124   
    E_u = np.zeros((T), dtype=np.float64)
    E_k = np.zeros((T), dtype=np.float64)
    for t in range(T):
        U_t = 0.0
        for i in range(N):
            E_k[t] = (1/(2*m))*np.sum(p[:,i,t]**2)
            for j in range(N):  
                if j != i: 
                    r_ij = np.sqrt(np.sum((q[:,i,t] - q[:,j,t])**2))
                    # if r_ij == 0:
                        # U_t += 10000000000
                        # r_ij = 1
                    B_ij_val = B_ij_s(q[:,:,t], i, j, N)[0]
    
                    pair_energy = V_R(r_ij) - B_ij_val * V_A(r_ij)
                    U_t += pair_energy
        E_u[t] = U_t/2
               
    # return np.abs(np.sum(E_u))
    return E_k, E_u

# solv_vec = np.zeros((6), dtype=np.float64)

# for i in range(len(solv_vec)):
#     solv_vec[i] = i
    
# print(solv_vec[:3])

# print(solv_vec[3:])
# N = 10
# T = 1000
# h = 0.1



# q_solv = np.zeros((3, N, T), dtype=np.float64)
# p_solv = np.zeros((3, N, T), dtype=np.float64)

# for i in range(N-1):
#     q_solv[:,i+1,0] = q_solv[:,i,0] + np.array([0,0,1.8], dtype=np.float64)
N = 40
# T = 25500
T = 1000

q_0 = np.zeros((3, N), dtype=np.float64)
# p_not = np.zeros((3, N), dtype=np.float64)

# for i in range(N-1):
#     # q_0[:,i+1] = q_0[:,i] + np.array([0,1.315,0], dtype=np.float64)
    
#     q_0[:,i+1] = q_0[:,i] + np.array([0,1.3233363844695725,0], dtype=np.float64)

# j = 0
# k = 0
# for i in range(N-1):
#     if k >5:
#         k = 0
#     q_0[:,i] = np.array([k+1.315,j+1.315,0], dtype=np.float64)
#     k+=1
#     if i%5:
#         j+=1
    
    
# q_0 = create_graphene_lattice(N=N)
# q_0 = create_carbon_ring(N=N, bond_length = 1.3233363844695725)
# print(q_0[:,3:9])
# q_0[:,-1] = np.array([0.41811706, 7.0006,3], dtype=np.float64)
# q_0[:,-1] = np.array([10, 15,0], dtype=np.float64)
# q_0 = np.array([[0.000, -0.698, -2.095, -2.794, -2.096, -0.698, -2.794/2],
#                 [0.000, 1.210, 1.210, 0.000, -1.210, -1.210, 0.000],
#                 [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 7.000]], dtype=np.float64)

q_00 = np.array([
    [-1.149, -1.149, -1.149, -1.149,  1.149,  1.149,  1.149,  1.149,  0.000,  0.000,
      0.000,  0.000,  1.859, -1.859,  1.859, -1.859,  0.710,  0.710, -0.710, -0.710],
    [-1.149, -1.149,  1.149,  1.149, -1.149, -1.149,  1.149,  1.149,  0.710,  0.710,
     -0.710, -0.710,  0.000,  0.000,  0.000,  0.000,  1.859, -1.859,  1.859, -1.859],
    [-1.149,  1.149, -1.149,  1.149, -1.149,  1.149, -1.149,  1.149,  1.859, -1.859,
      1.859, -1.859,  0.710,  0.710, -0.710, -0.710,  0.000,  0.000,  0.000,  0.000]
])

# q_0 = np.array([[0.000, 1.315, 0  ],
#                 [0.000, 1.0,1.315 ],
#                 [0.000, 0.000, 0.000 ]], dtype=np.float64)

# q_0 = np.array([[0.000, 0.94019 , -0.929012  ],
#                 [0.000, -0.931269, 0.942418 ],
#                 [0.000, 0.000, 0.000 ]], dtype=np.float64)

# q_0 = np.array([[0.000, 1.225489 , 1.526593, -0.817709  ],
#                 [0.000, 0.188804 , -1.15727,-1.119067 ],
#                 [0.000, 0.000, 0.000, 0.0 ]], dtype=np.float64)

# print(np.rad2deg(np.arccos(np.sum(q_0[:,1]*q_0[:,2])/(np.linalg.norm(q_0[:,1])*np.linalg.norm(q_0[:,2])))))
# print(np.rad2deg(np.arccos(np.sum(q_0[:,2]*q_0[:,3])/(np.linalg.norm(q_0[:,2])*np.linalg.norm(q_0[:,3])))))

# два кольца взаимодействут
NN = int(N/2)
# q_01 = create_carbon_ring(N=NN, bond_length = 1.3233363844695725)
# q_02 = create_carbon_ring(N=NN, bond_length = 1.3233363844695725)

# q_02[0,:] = q_02[0,:] + 14.5
# q_02[1,:] = q_02[1,:] + 0.5

# q_0[0,:NN] = q_01[0,:]
# q_0[1,:NN] = q_01[1,:]
# q_0[2,:NN] = q_01[2,:]

# q_0[0, NN:] = q_02[0,:]
# q_0[1, NN:] = q_02[1,:]
# q_0[2, NN:] = q_02[2,:]

q_01 = q_00
q_02 = q_00 

q_0[0,:NN] = q_01[0,:]
q_0[1,:NN] = q_01[1,:]
q_0[2,:NN] = q_01[2,:]

q_0[0, NN:] = q_02[0,:] + 7.5
q_0[1, NN:] = q_02[1,:] + 3.7 
q_0[2, NN:] = q_02[2,:]

@nb.njit(parallel=True)
def calc_main(q_not, N, T):
    h = 0.1

    q_solv = np.zeros((3, N, T), dtype=np.float64)
    p_solv = np.zeros((3, N, T), dtype=np.float64)
    
    # p_not = np.random.randn(3, N)*10
    # q_not = np.random.randn(3, N)*2.5
    # p_solv[:,0,0] = np.array([0.94019 ,-0.931269,0.0], dtype=np.float64)*10
    # p_solv[:,-1,0] = np.array([0.0,0.0,0.01], dtype=np.float64)
    # p_solv[:,-3,0] = np.array([0,0,-100.0], dtype=np.float64)
    # p_solv[:,10,0] = np.array([0.0001,0.0,0.0], dtype=np.float64)
    for m in range(int(N/2),N):
        p_solv[0,m,0] = -3.0
    
    q_solv[:,:,0] = q_not
    # p_solv[:,:,0] = p_not
    
    # for t in range(1,T):
    #     for i in nb.prange(N):
            # p_solv[:,i,t], q_solv[:,i,t] = runge_four(q_solv[:,:,t-1], p_solv[:,i,t-1], i, h, N)
            # p_solv[:,i,t], q_solv[:,i,t] = symplectic_integrator_6th_order(q_solv[:,:,t-1], p_solv[:,i,t-1], i, h, N)
    for t in range(1,T):
        p_solv[:,:,t], q_solv[:,:,t] = runge_our(q_solv[:,:,t-1], p_solv[:,:,t-1], h, N)
        # p_solv[:,:,t], q_solv[:,:,t] = yoshida_four(q_solv[:,:,t-1], p_solv[:,:,t-1], h, N)

    return q_solv, p_solv

start_time = time.perf_counter()
# print(f"start_time: {start_time:.6f} ")

q, p = calc_main(q_0, N, T)

end_time = time.perf_counter()
# print(f"end_time: {end_time:.6f}" )
execution_time = end_time - start_time
# print(q)
print(f"Время выполнения: {execution_time:.6f} секунд")

# plt.plot(q[0,0,:])
# plt.plot(q[1,0,:])
# plt.plot(q[2,0,:])
# plt.show()

# plt.plot(q[0,0,:], q[1,0,:])
# plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])   

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

# ax.set_xlim(np.min(q[0]), np.max(q[0]))
# ax.set_ylim(np.min(q[1]), np.max(q[1]))
# ax.set_zlim(np.min(q[2]), np.max(q[2]))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Молекулярная динамика')

scat = ax.scatter([], [], [], s=20, c='blue', alpha=0.7)

def animate(frame):
    """Обновление данных для каждого кадра анимации"""
    x = q[0, :, frame]
    y = q[1, :, frame]
    z = q[2, :, frame]
    
    # Обновляем данные точек
    scat._offsets3d = (x, y, z)
    
    # Обновляем заголовок с номером кадра
    ax.set_title(f'Молекулярная динамика, шаг: {frame}/{T}')
    return scat,

# Создаем анимацию
ani = animation.FuncAnimation(
    fig=fig,
    func=animate,
    frames=T,
    interval=0.1,  
    blit=False,
    repeat=True
)

plt.show()


# E_k, E_u = energy(p, q, N, T)

# plt.plot(E_k)
# plt.show()
# plt.plot(E_u)
# plt.show()
# plt.plot(E_k+E_u)
# plt.show()

# def plot_system(vec_q, forces=None, scale=0.3):
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     atoms = vec_q.T
#     ax.scatter(atoms[:,0], atoms[:,1], atoms[:,2], c='red', s=100, label='Атомы')
    
#     if forces is not None:
#         for i in range(vec_q.shape[1]):
#             ax.quiver(atoms[i,0], atoms[i,1], atoms[i,2], 
#                      forces[0,i], forces[1,i], forces[2,i],
#                      length=scale, color='blue', alpha=0.7, 
#                      label='Силы' if i == 0 else "")
    
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y') 
#     ax.set_zlabel('Z')
#     ax.legend()
#     plt.title('Конфигурация атомов и силы')
#     plt.show()
    
# print("\nВизуализация системы из 4 атомов:")
# vec_q_test = np.array([[0.0, 0.0, 0.0],
#                         [1.5, 0.0, 0.0], 
#                         [0.75, 1.3, 0.0],
#                         [0.75, 0.433, 1.2]]).T

# forces_test = np.zeros((3, 4))
# forces_test_sum = np.zeros((3))
# for n in range(4):
#     forces_test[:, n] = dHdq(vec_q_test, n, 4)
#     forces_test_sum += forces_test[:, n]
# print(forces_test_sum)
# plot_system(vec_q_test, forces_test)





# fig, ax = plt.subplots()

# # Определяем функцию, которая будет вызываться на каждом кадре анимации
# def update(frame):
#     # Очищаем предыдущий кадр
#     ax.clear()
#     # Рисуем новый кадр
#     ax.plot(q[0, :, frame], label='1')
#     # ax.plot(q[1, :, frame], label='2')
 
#     ax.set_title("Frame {}".format(frame))
#     ax.legend()
#     # ax.set_xlim(xx.min(), xx.max()) 
#     # ax.set_ylim(0, 1e-9)

# # Создаем анимацию
# ani = animation.FuncAnimation(fig, update, frames=T, interval=0.1)
# # ani.save("elbrus.gif", writer='pillow', fps=60)
# # Показываем анимацию
# plt.show()

# fig, ax = plt.subplots()

# # Определяем функцию, которая будет вызываться на каждом кадре анимации
# def update(frame):
#     # Очищаем предыдущий кадр
#     ax.clear()
#     # Рисуем новый кадр
#     # ax.plot(np.diff(q[1, :, frame]), label='1')
#     ax.plot((q[2, :, frame]), label='1')
#     # ax.plot(q[1, :, frame], label='2')
 
#     ax.set_title("Frame {}".format(frame))
#     ax.legend()
#     # ax.set_xlim(xx.min(), xx.max()) 
#     # ax.set_ylim(0, 1e-9)

# # Создаем анимацию
# ani = animation.FuncAnimation(fig, update, frames=T, interval=0.1)
# # ani.save("elbrus.gif", writer='pillow', fps=60)
# # Показываем анимацию
# plt.show()