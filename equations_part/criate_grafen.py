import numpy as np

def create_graphene_lattice(N, a=1.42, box_size=None):
    """
    Создает графеновую решетку для молекулярной динамики
    
    Parameters:
    -----------
    N : int
        Общее количество атомов (должно быть четным для симметричной решетки)
    a : float
        Длина связи углерод-углерод в ангстремах (1.42 Å для графена)
    box_size : tuple
        Размеры расчетной ячейки (Lx, Ly, Lz)
    
    Returns:
    --------
    positions : ndarray
        Массив координат формы (3, N)
    """
    
    # Параметры графеновой решетки
    a_cc = a  # длина связи C-C
    a1 = a_cc * np.array([np.sqrt(3), 0.0, 0.0])  # базисный вектор 1
    a2 = a_cc * np.array([np.sqrt(3)/2, 3.0/2, 0.0])  # базисный вектор 2
    
    # Определяем размеры решетки
    n_x = int(np.sqrt(N/2)) + 1
    n_y = int(np.sqrt(N/2)) + 1
    
    positions = []
    
    # Создаем позиции атомов в гексагональной решетке
    for i in range(n_x):
        for j in range(n_y):
            # Атом типа A в графеновой решетке
            pos_A = i * a1 + j * a2
            positions.append(pos_A)
            
            # Атом типа B в графеновой решетке (смещенный)
            pos_B = i * a1 + j * a2 + a_cc * np.array([0.0, 1.0, 0.0])
            positions.append(pos_B)
            
            # Если достигли нужного количества атомов, выходим
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    
    # Обрезаем до нужного количества атомов
    positions = positions[:N]
    positions = np.array(positions).T  # преобразуем к форме (3, N)
    
    # Центрируем решетку
    center = np.mean(positions, axis=1, keepdims=True)
    positions -= center
    
    # Если задан размер ячейки, масштабируем позиции
    if box_size is not None:
        Lx, Ly, Lz = box_size
        current_size = np.max(positions, axis=1) - np.min(positions, axis=1)
        scale_x = Lx / (current_size[0] + a_cc)
        scale_y = Ly / (current_size[1] + a_cc)
        positions[0] *= scale_x
        positions[1] *= scale_y
        positions[2] = 0  # все атомы в плоскости z=0
    
    return positions


def create_carbon_ring(N, bond_length=1.315):
    """
    Создает равновесную замкнутую углеродную цепочку (кольцо)
    
    Parameters:
    -----------
    N : int
        Количество атомов углерода
    bond_length : float
        Длина связи в Å (по умолчанию 1.315 Å)
    
    Returns:
    --------
    q_0 : np.ndarray
        Массив координат размером (3, N)
    """
    # Вычисляем радиус кольца
    # Для правильного N-угольника с длиной стороны bond_length:
    # bond_length = 2 * R * sin(π/N)
    # => R = bond_length / (2 * sin(π/N))
    R = bond_length / (2 * np.sin(np.pi / N))
    
    # Создаем массив для координат
    q_0 = np.zeros((3, N), dtype=np.float64)
    
    # Заполняем координаты атомов в плоскости XY
    for i in range(N):
        angle = 2 * np.pi * i / N  # Угол для i-го атома
        
        # Координаты в плоскости XY, Z=0
        q_0[0, i] = R * np.cos(angle)  # X координата
        q_0[1, i] = R * np.sin(angle)  # Y координата
        q_0[2, i] = 0.0                # Z координата
    
    return q_0