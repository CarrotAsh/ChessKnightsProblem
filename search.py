import numpy as np

def initial_state(M, N):
    # Crea un tablero vacío usando 0s
    return np.zeros((M, N), dtype=int)

# Ejemplo de uso de la función estado inicial
def expand(board):
    boards = [] # Crea una lista vacía de tableros

    # Crea una lista de tableros con todas las posibles jugadas
    def copiar_tablero(board):
          return np.copy(board)

    def movimientos_de_caballo(M, N, x, y):
          movimientos = [(x + 2, y + 1), (x + 2, y - 1), (x - 2, y + 1), (x - 2, y - 1), (x + 1, y + 2), (x + 1, y - 2), (x - 1, y + 2), (x - 1, y - 2) ]
          return [(i, j) for i, j in movimientos if 0 <= i < M and 0 <= j < N]

    def colocar_caballo(board, x, y):
          board[x][y] = 1
          return board

    return boards # Devolvemos una lista de tableros

# Pistas:
# - Una función que copie un tablero completo
# - Una función que coloque un caballo en una posición dada en i, j
# - Una estructura de datos con los movimientos posibles para un caballo
# Pistas:
# - Una función que copie un tablero completo
# - Una función que coloque un caballo en una posición dada en i, j
# - Una estructura de datos con los movimientos posibles para un caballo

expand(board) # Debe devolver una lista de tableros

def is_solution(board):
    # Verifica si un tablero es solución
    sol = None

    # Haz todas las comprobaciones necesarias para determinar
    # si el tablero es solución

    return sol # Devuelve True si es solución, False en caso contrario

def cost(path): # path debe contener VARIOS tableros
    # Calcula el coste de un camino
    # Esto debería ser posible: board = path[-1]
    cost = 0

    # Calcula el coste de un camino completo

    return cost

# Pista:
# - Recuerda que A* y B&B funcionan minimizando el coste.
# - ¿Podemos afrontar este problema de otra manera? Maximizar las casillas ocupadas NO funciona...

def cost(path): # path debe contener VARIOS tableros
    # Calcula el coste de un camino
    # Esto debería ser posible: board = path[-1]
    cost = 0

    # Calcula el coste de un camino completo

    return cost

# Pista:
# - Recuerda que A* y B&B funcionan minimizando el coste.
# - ¿Podemos afrontar este problema de otra manera? Maximizar las casillas ocupadas NO funciona...

def prune(path_list):
    # Si detecta que dos caminos llevan al mismo estado,
    # solo nos interesa aquel camino de menor coste
    # Más adelante usamos la poda despues de ordenar
    return [] # Devuelve una lista de caminos

# *args y **kwargs son argumentos variables, si el argumento no es reconocido es almacenado en estas variables.
# Aquí se utilizan para ignorar argumentos innecesarios.

def order_astar(old_paths, new_paths, c, h, *args, **kwargs):
    # Ordena la lista de caminos según una heurística
    return prune([]) # Devuelve la lista de caminos ordenada y podada segun A*

def order_byb(old_paths, new_paths, c, *args, **kwargs):
    # Ordena la lista de caminos según una heurística
    return prune([]) # Devuelve la lista de caminos ordenada y podada segun B&B

def search(initial_board, expansion, cost, heuristic, ordering, solution):
    # Realiza una búsqueda en el espacio de estados
    paths = None # Crea la lista de caminos
    sol = None # Este es el estado solucion

    # 1 - Mientras haya caminos y no se haya encontrado solución
    # 2 - Extraer el primer camino
    # 3 - Comprobar si estamos ante un estado solución
    # 4 - Si no es solución, expandir el camino/ Si es solucion, detenemos y vamos al paso 7.
    # 5 - Para cada estado expandido nuevo, añadirlo al camino lo que genera una lista de nuevos caminos
    # 6 - Ordenar los nuevos caminos y viejos caminos, y realizar poda. Volver al paso 1.
    # 7 - Devolver el camino si es solución, si no devolver None

    if len(paths) > 0:
        return sol # Devuelve solo la solucion, no el camino solucion
    else:
        return None

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