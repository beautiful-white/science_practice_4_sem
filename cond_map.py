import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from itertools import combinations
from math import sqrt

# CONST -------
V = 0.3  # Скорость света, км/мкс

diff_pairs = np.array([np.array([2, 1]), np.array([3, 2]),
                       np.array([4, 3]), np.array([3, 1]),
                       np.array([4, 1]), np.array([4, 2]),
                       np.array([1, 2]), np.array([1, 3]),
                       np.array([1, 4]), np.array([2, 3]),
                       np.array([2, 4]), np.array([3, 4])], dtype=np.int8)
# -------------


def calc_distance(xy1, xy2):
    return sqrt((xy1[0]-xy2[0])**2+(xy1[1]-xy2[1])**2)


def calc_time_of_arrival(d, speed):
    return d / speed


def get_cond(a1, a2, a3, a4, RFsource=(1, 1), haserror=False, error_matrix=np.zeros(9).reshape(3, 3)):
    a_arr = np.array([a1, a2, a3, a4])
    r_arr = np.array([])
    t1 = calc_time_of_arrival(calc_distance(a1, RFsource), V)
    t2 = calc_time_of_arrival(calc_distance(a2, RFsource), V)
    t3 = calc_time_of_arrival(calc_distance(a3, RFsource), V)
    t4 = calc_time_of_arrival(calc_distance(a4, RFsource), V)
    t_arr = np.array([t1, t2, t3, t4], dtype=np.float64)
    # print("Антенны (км):\n", a_arr)
    # print("Времена прибытия (мкс):\n", t_arr)

    g = []
    for i in diff_pairs:
        a_i, a_j, t_i, t_j = a_arr[i[0] -
                                   1], a_arr[i[1]-1], t_arr[i[0]-1], t_arr[i[1]-1]
        g_ij = ((a_i[0] ** 2 + a_i[1] ** 2 - V ** 2 * t_i ** 2) -
                (a_j[0] ** 2 + a_j[1] ** 2 - V ** 2 * t_j ** 2))/2
        g.append(g_ij)
    g = np.array(g, dtype=np.float64)
    # print("Вектор g:\n", g)

    all_matrix = []
    all_matrix_g = []
    for i in combinations(diff_pairs, 3):
        diff_index = []
        diff_index_list = diff_pairs.tolist()
        diff_index.append(g[diff_index_list.index(i[0].tolist())])
        diff_index.append(g[diff_index_list.index(i[1].tolist())])
        diff_index.append(g[diff_index_list.index(i[2].tolist())])
        matrix = np.array([[a_arr[i[0][0]-1][0] - a_arr[i[0][1]-1][0],
                            a_arr[i[0][0]-1][1] - a_arr[i[0][1]-1][1],
                            V**2 * (t_arr[i[0][0]-1] - t_arr[i[0][1]-1])],
                           [a_arr[i[1][0]-1][0] - a_arr[i[1][1]-1][0],
                            a_arr[i[1][0]-1][1] - a_arr[i[1][1]-1][1],
                            V**2 * (t_arr[i[1][0]-1] - t_arr[i[1][1]-1])],
                           [a_arr[i[2][0]-1][0] - a_arr[i[2][1]-1][0],
                            a_arr[i[2][0]-1][1] - a_arr[i[2][1]-1][1],
                            V**2 * (t_arr[i[2][0]-1] - t_arr[i[2][1]-1])]], dtype=np.float64)
        matrix = matrix + error_matrix
        all_matrix.append(matrix)
        all_matrix_g.append(np.array(diff_index))
    all_matrix = np.array(all_matrix)
    # print("Матрицы K:\n", all_matrix)

    all_matrix_cond = []
    for matrix in all_matrix:
        all_matrix_cond.append(linalg.cond(matrix))
    min_cond = min(all_matrix_cond)
    return min_cond


def main():
    # Антенны в километрах
    a1 = np.array([-126.9813707,  -33.20465762])
    a2 = np.array([-115.815395,   104.6189058])
    a3 = np.array([65.14195069,  -42.17090147])
    a4 = np.array([-61.85096707, - 67.58373789])
    a_arr = np.array([a1, a2, a3, a4])

    # Сетка в километрах
    x = np.arange(-300, 300, 10)
    y = np.arange(-300, 300, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    hasError = True

    error_matrix = np.zeros(9).reshape(3, 3)
    if hasError:
        error_matrix = [[0, 0,
                         np.random.randint(-1, 1)]for i in range(3)]
        error_matrix = np.array(error_matrix, dtype=np.float64)

    # Вычисление минимального числа обусловленности для каждой точки
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = get_cond(a1, a2, a3, np.array(
                [X[i, j], Y[i, j]]), error_matrix=error_matrix)

    # Построение контурной карты
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, np.log10(Z), levels=20, cmap='viridis')
    plt.colorbar(
        contour, label='Логарифм минимального числа обусловленности (log10)')
    plt.scatter(a_arr[:, 0], a_arr[:, 1], c='blue', label='Антенны', s=50)
    plt.title('Карта чисел обусловленности')
    plt.xlabel('X (км)')
    plt.ylabel('Y (км)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()
