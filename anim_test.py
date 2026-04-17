import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Ваши данные (замените на реальные)
# q_solv = np.load('your_data.npy')  # или напрямую из вашего кода
N = 100   # количество частиц
T = 200   # количество временных шагов
q_solv = np.random.randn(3, N, T) * 10  # пример случайных данных
# q_solv[:,:,0] = create_graphene_lattice(N, a=1.42, box_size=None)

# Создаем фигуру и 3D оси
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])  # одинаковый масштаб по осям

# Настройка осей
ax.set_xlim(np.min(q_solv[0]), np.max(q_solv[0]))
ax.set_ylim(np.min(q_solv[1]), np.max(q_solv[1]))
ax.set_zlim(np.min(q_solv[2]), np.max(q_solv[2]))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Молекулярная динамика')

# Инициализация scatter plot
scat = ax.scatter([], [], [], s=20, c='blue', alpha=0.7)

def animate(frame):
    """Обновление данных для каждого кадра анимации"""
    x = q_solv[0, :, 0]
    y = q_solv[1, :, 0]
    z = q_solv[2, :, 0]
    
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
    interval=50,  # интервал между кадрами в мс
    blit=False,
    repeat=True
)

plt.show()