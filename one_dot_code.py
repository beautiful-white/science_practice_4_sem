import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from itertools import combinations
from math import sqrt


def main():
    a1 = np.array([-10., -10.])
    a2 = np.array([10., -10.])
    a3 = np.array([10., 10.])
    a4 = np.array([-10., 10.])
    a_arr = np.array([a1, a2, a3, a4])

    RFsource = (56, 42)

    # CONST -------
    V = 0.3  # Скорость света, м/с

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

    t1 = calc_time_of_arrival(calc_distance(a1, RFsource), V)
    t2 = calc_time_of_arrival(calc_distance(a2, RFsource), V)
    t3 = calc_time_of_arrival(calc_distance(a3, RFsource), V)
    t4 = calc_time_of_arrival(calc_distance(a4, RFsource), V)
    t_arr = np.array([t1, t2, t3, t4], dtype=np.float64)
    print("Антенны:\n", a_arr)
    print("Времена прибытия (с):\n", t_arr)

    g = []
    for i in diff_pairs:
        a_i, a_j, t_i, t_j = a_arr[i[0] -
                                   1], a_arr[i[1]-1], t_arr[i[0]-1], t_arr[i[1]-1]
        g_ij = ((a_i[0] ** 2 + a_i[1] ** 2 - V ** 2 * t_i ** 2) -
                (a_j[0] ** 2 + a_j[1] ** 2 - V ** 2 * t_j ** 2))/2
        g.append(g_ij)
    g = np.array(g, dtype=np.float64)
    print("Вектор g:\n", g)

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
        all_matrix.append(matrix)
        all_matrix_g.append(np.array(diff_index))
    all_matrix = np.array(all_matrix)
    print("Матрицы K:\n", all_matrix)

    all_matrix_cond = []
    for matrix in all_matrix:
        all_matrix_cond.append(linalg.cond(matrix))
    min_cond = min(all_matrix_cond)

    print("Числа обусловленности:\n", all_matrix_cond)

    index = all_matrix_cond.index(min_cond)
    print("Минимальное число обусловленности:", min_cond, "Индекс:", index)

    main_matrix = all_matrix[index]
    main_g = all_matrix_g[index]
    print("Лучшая матрица K:\n", main_matrix)
    print("Соответствующий вектор g:\n", main_g)

    # Решение через псевдообратную матрицу
    solution = np.dot(linalg.pinv(main_matrix), main_g)
    x, y, vt = solution
    print("Решение: x=", x, "y=", y, "vt=", vt)

    print("Оценённое положение: x={:.4f}, y={:.4f}".format(x, y))
    error = calc_distance((x, y), RFsource)
    print("ОШИБКА: {:.4f} МЕТРОВ".format(error))

    # Визуализация
    plt.figure(figsize=(6, 6))
    plt.scatter(a_arr[:, 0], a_arr[:, 1], c='blue', label='Антенны', s=50)
    plt.scatter(RFsource[0], RFsource[1], c='red',
                label='Истинный источник', s=50)
    plt.scatter(x, y, c='green', label='Оценённый источник', s=50)
    plt.title('Положение антенн и источника')
    plt.xlabel('X (км)')
    plt.ylabel('Y (км)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()
