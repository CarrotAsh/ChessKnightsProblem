import numpy as np

def initial_state(M, N):
    return np.zeros((M, N), dtype=int)

board = initial_state(3, 3)
print(board)

def copy_board(board):
    return np.copy(board)

def knight_movements(M, N, x, y):
    movements = [(x + 2, y + 1), (x + 2, y - 1), (x - 2, y + 1), (x - 2, y - 1), (x + 1, y + 2), (x + 1, y - 2),
                   (x - 1, y + 2), (x - 1, y - 2)]
    return [(i, j) for i, j in movements if 0 <= i < M and 0 <= j < N]

def place_knight(board, x, y):
    board[x][y] = 1

    for i, j in knight_movements(board.shape[0], board.shape[1], x, y):
        board[i][j] = -1

    return board

def expand(board):
    boards = []

    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i][j] == 0:
                new_board = copy_board(board)
                place_knight(new_board, i, j)
                boards.append(new_board)

    return boards

def is_solution(board):
    return not np.any(board == 0)

def cost(path):
    board = path[-1]
    return np.sum(board[board == -1])

def heuristic_1(board):
    checkerboard_mask = (np.indices(board.shape).sum(axis=0) % 2 == 1)
    threatened_squares_mask = (board == -1)
    black_threatened_squares = np.sum(threatened_squares_mask & checkerboard_mask)
    num_knights = np.sum(board == 1)

    return board.shape[0] * board.shape[1] - num_knights - black_threatened_squares

def prune(path_list):
    unique_paths = []
    for path in path_list:
        unique = True
        for uniq in unique_paths:
            if np.array_equal(path[-1], uniq[-1]):
                unique = False
                break
        if unique:
            unique_paths.append(path)

    return unique_paths

def order_astar(old_paths, new_paths, c, h, *args, **kwargs):
    # Ordena la lista de caminos según una heurística
    old_paths.extend(new_paths)
    old_paths = prune(old_paths)
    old_paths.sort(key= lambda x: h(x[-1]))

    return old_paths  # Devuelve la lista de caminos ordenada y podada segun A*

def order_byb(old_paths, new_paths, c, *args, **kwargs):
    # Ordena la lista de caminos según una heurística
    return prune([]) # Devuelve la lista de caminos ordenada y podada segun B&B

def search(initial_board, expansion, cost, heuristic, ordering, solution):
    # Realiza una búsqueda en el espacio de estados
    paths = [initial_board] # Crea la lista de caminos
    sol = None # Este es el estado solucion

    while paths and sol is None:
        path = paths[0]
        if is_solution(path):
            sol = path
            break
        new_paths = expand(path)
        if heuristic is None:
            paths = order_byb(paths, new_paths, cost)
        else:
            paths = order_astar(paths, new_paths, cost, heuristic)

        print("Number of paths remaining: ", len(paths))
        print(board)

    if len(paths) > 0:
        return sol  # Devuelve solo la solucion, no el camino solucion
    else:
        return None

    # 1 - Mientras haya caminos y no se haya encontrado solución
    # 2 - Extraer el primer camino
    # 3 - Comprobar si estamos ante un estado solución
    # 4 - Si no es solución, expandir el camino/ Si es solucion, detenemos y vamos al paso 7.
    # 5 - Para cada estado expandido nuevo, añadirlo al camino lo que genera una lista de nuevos caminos
    # 6 - Ordenar los nuevos caminos y viejos caminos, y realizar poda. Volver al paso 1.
    # 7 - Devolver el camino si es solución, si no devolver None

'''

    ################################# NO TOCAR #################################
    #                                                                          #
    import time

    def timer(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            print("Executime time: ", end - start, " seconds")
            return res

        return wrapper
    #                                                                          #
    ################################# NO TOCAR #################################

    # Este codigo temporiza la ejecución de una función cualquiera

    ################################# NO TOCAR #################################
    #                                                                          #
    @timer
    def search_horse_byb(initial_board):
        return search(initial_board, expand, cost, None, order_byb, is_solution)

    @timer
    def search_horse_astar(initial_board, heuristic):
        return search(initial_board, expand, cost, heuristic, order_astar, is_solution)
    #                                                                          #
    ################################# NO TOCAR #################################

CONF = {'2x2': (2, 2),
        '3x3': (3, 3),
        '3x5': (3, 5),
        '5x5': (5, 5),
        '8x8': (8, 8),
        }

def measure_solution(board):
    # Devuelve el número de caballos en la solución
    # Es necesario programarla para poder medir la calidad de la solución
    return 0

def launch_experiment(configuration, heuristic=None):
    conf = CONF[configuration]
    print(f"Running {'A*' if heuristic else 'B&B'} with {configuration} board")
    if heuristic:
        sol = search_horse_astar(initial_state(*conf), heuristic)
    else:
        sol = search_horse_byb(initial_state(*conf))
    n_c = measure_solution(sol)
    print(f"Solution found: \n{sol}")
    print(f"Number of horses in solution: {n_c}")

    return sol, n_c

launch_experiment('2x2') # Ejemplo de uso para B&B
print()
launch_experiment('3x5', heuristic=heuristic_1) # Ejemplo de uso para A*
print("Execution finished")

### Coloca aquí tus experimentos ###

### Coloca aquí tus experimentos ###
'''
