import numpy as np
import numba as nb
import matplotlib.pyplot as plt

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
def V_R(r):
    D_e = 6.325
    S = 1.29
    beta = 1.5
    R_e = 1.315
    return f_cut_off(r)*(D_e/(S-1))*np.exp(-beta*np.sqrt(2*S)*(r-R_e))

@nb.njit
def V_A(r):
    D_e = 6.325
    S = 1.29
    beta = 1.5
    R_e = 1.315
    return f_cut_off(r)*(D_e*S/(S-1))*np.exp(-beta*np.sqrt(2/S)*(r-R_e))

N = 100
r = np.linspace(0,5, N)
# r = np.linspace(0,20e-10, N)
ra = np.zeros((N), dtype=np.float64)
rv = np.zeros((N), dtype=np.float64)
for i in range(N):
    rv[i] = V_R(r[i])
    ra[i] = V_A(r[i])
    
# plt.plot(r, ra)
# plt.plot(r, rv)

plt.plot(r, rv-ra)
plt.show()

plt.plot(r[1:], -np.diff(rv-ra)/(r[1]-r[0]))
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Параметры системы
N = 100     # Количество точек в пространстве
T = 200     # Количество временных шагов

# Создаем пустой массив и заполняем данными (пример с бегущей волной)
data = np.zeros([N, T])
for t in range(T):
    data[:, t] = np.sin(2 * np.pi * (0.1 * t + 5 * np.linspace(0, 1, N)))

# Создаем фигуру и оси
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(N)
line, = ax.plot(x, data[:, 0], 'b-', linewidth=2)
ax.set_ylim([-1.1, 1.1])
ax.set_xlabel('Пространство (x)')
ax.set_ylabel('Амплитуда')
ax.grid(True)

# Функция инициализации анимации
def init():
    line.set_ydata(data[:, 0])
    return line,

# Функция обновления кадра
def update(frame):
    line.set_ydata(data[:, frame])
    ax.set_title(f'Временной шаг: {frame}/{T}')
    return line,

# Создаем анимацию
ani = FuncAnimation(
    fig,
    update,
    frames=T,
    init_func=init,
    blit=True,
    interval=50,
    repeat=True
)

plt.tight_layout()
plt.show()

# Для сохранения анимации в файл (раскомментируйте при необходимости)
# ani.save('wave_animation.gif', writer='pillow', fps=20)