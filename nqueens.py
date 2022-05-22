import sys
import random
from math import e

MAXQ = 100


def in_conflict(column, row, other_column, other_row):
    """
    Checks if two locations are in conflict with each other.
    :param column: Column of queen 1.
    :param row: Row of queen 1.
    :param other_column: Column of queen 2.
    :param other_row: Row of queen 2.
    :return: True if the queens are in conflict, else False.
    """
    if column == other_column:
        return True  # Same column
    if row == other_row:
        return True  # Same row
    if abs(column - other_column) == abs(row - other_row):
        return True  # Diagonal

    return False


def in_conflict_with_another_queen(row, column, board):
    """
    Checks if the given row and column correspond to a queen that is in conflict with another queen.
    :param row: Row of the queen to be checked.
    :param column: Column of the queen to be checked.
    :param board: Board with all the queens.
    :return: True if the queen is in conflict, else False.
    """
    for other_column, other_row in enumerate(board):
        if in_conflict(column, row, other_column, other_row):
            if row != other_row or column != other_column:
                return True
    return False


def count_conflicts(board):
    """
    Counts the number of queens in conflict with each other.
    :param board: The board with all the queens on it.
    :return: The number of conflicts.
    """
    cnt = 0

    for queen in range(0, len(board)):
        for other_queen in range(queen + 1, len(board)):
            if in_conflict(queen, board[queen], other_queen, board[other_queen]):
                cnt += 1

    return cnt


def evaluate_state(board):
    """
    Evaluation function. The maximal number of queens in conflict can be 1 + 2 + 3 + 4 + .. +
    (nquees-1) = (nqueens-1)*nqueens/2. Since we want to do ascending local searches, the evaluation function returns
    (nqueens-1)*nqueens/2 - countConflicts().

    :param board: list/array representation of columns and the row of the queen on that column
    :return: evaluation score
    """
    return (len(board) - 1) * len(board) / 2 - count_conflicts(board)


def print_board(board):
    """
    Prints the board in a human readable format in the terminal.
    :param board: The board with all the queens.
    """
    print("\n")

    for row in range(len(board)):
        line = ''
        for column in range(len(board)):
            if board[column] == row:
                line += 'Q' if in_conflict_with_another_queen(row, column, board) else 'q'
            else:
                line += '.'
        print(line)


def init_board(nqueens):
    """
    :param nqueens integer for the number of queens on the board
    :returns list/array representation of columns and the row of the queen on that column
    """

    board = []

    for column in range(nqueens):
        board.append(random.randint(0, nqueens - 1))

    return board


"""
------------------ Do not change the code above! ------------------
"""


def random_search(board):
    """
    This function is an example and not an efficient solution to the nqueens problem. What it essentially does is flip
    over the board and put all the queens on a random position.
    :param board: list/array representation of columns and the row of the queen on that column
    """

    i = 0
    optimum = (len(board) - 1) * len(board) / 2

    while evaluate_state(board) != optimum:
        i += 1
        print('iteration ' + str(i) + ': evaluation = ' + str(evaluate_state(board)))
        if i == 1000:  # Give up after 1000 tries.
            break

        for column, row in enumerate(board):  # For each column, place the queen in a random row
            board[column] = random.randint(0, len(board) - 1)

    if evaluate_state(board) == optimum:
        print('Solved puzzle!')

    print('Final state is:')
    print_board(board)


def hill_climbing(board):
    """
    Implement this yourself.
    :param board:
    :return:
    """
    i = 0
    optimum = (len(board) - 1) * len(board) / 2
    best_neighbour = board.copy()
    neighbour = best_neighbour.copy()
    neighbour = board.copy()
    while evaluate_state(best_neighbour) != optimum:
        i += 1
        print('iteration ' + str(i) + ': evaluation = ' + str(evaluate_state(best_neighbour)))
        if i > 1000:  # Give up after 1000 tries.
            break
        for column, row in enumerate(neighbour):  # For each column
            for qrow in range(0, len(board)):
                if row == qrow:
                    continue
                neighbour[column] = qrow
                if evaluate_state(neighbour) > evaluate_state(best_neighbour):
                    best_neighbour = neighbour.copy()
                if evaluate_state(neighbour) == evaluate_state(best_neighbour):
                    best_neighbour = random.choice([best_neighbour, neighbour])
                neighbour = board.copy()
        board = best_neighbour.copy()
        neighbour = best_neighbour.copy()


    if evaluate_state(best_neighbour) == optimum:
        print('Solved puzzle!')

    print('Final state is:')
    print_board(best_neighbour)


def time_to_temperature(time):
    return 0.5*(e**(-time/7000))


def simulated_annealing(board):
    """
    Implement this yourself.
    :param board:
    :return:
    """
    t = 0
    optimum = (len(board) - 1) * len(board) / 2
    current = board.copy()
    while evaluate_state(board) != optimum:
        t += 1
        T = time_to_temperature(t)
        print('iteration ' + str(t) + ': evaluation = ' + str(evaluate_state(board)))
        random_column = random.randint(0, len(board)-1)
        board[random_column] = random.choice(list(range(0, board[random_column])) + list(range(board[random_column], len(board))))
        E = evaluate_state(board) - evaluate_state(current)
        if E > 0:
            current = board.copy()
        else:
            current = random.choices(population=[current, board], weights=[1 - e**(E/T), e**(E/T)]).copy()

    if evaluate_state(board) == optimum:
        print('Solved puzzle!')

    print('Final state is:')
    print_board(board)


def reproduce(parent1, parent2):
    """
    :param parent1: a board configuration
    :param parent2: a board configuration
    :return: a board configuration that concatenates a part of parent1 and 2
    """
    column = random.randint(0, len(parent2) - 1)
    child = parent1[0: column] + parent2[column: len(parent2)]
    return child


def mutate(child):
    """
    :param child: a board configurations
    :return: the child with one randomized value on a randomized index
    """
    mutated = child.copy()
    mutation_probability = 0.01
    column = random.randint(0, len(mutated) - 1)
    range1 = list(range(0, mutated[column]))
    range2 = list(range(mutated[column] + 1, len(mutated)))
    range1 = range1 + range2
    mutated[column] = random.choice(range1)
    return random.choices([child, mutated], weights=[1-mutation_probability, mutation_probability])[0]


def calculate_fitness(board):
    """
    :param board: an nqueens configuration
    :return: fitness of this configuration expressed in the number of non attack pairs
    """
    max_no_attacking_pairs = (len(board) * (len(board) - 1)) / 2
    no_attacking_pairs = max_no_attacking_pairs - count_conflicts(board)
    return no_attacking_pairs


def calculate_weights(population):
    """
    :param population: a list of board configurations
    :return: a list of weights based on fitness
    """
    best = []
    weights = []
    best_fitness = 0
    for individual in population:
        fitness = calculate_fitness(individual)
        if fitness > best_fitness:
            best = individual
        weights.append(fitness)
    return weights, best.copy()


def make_population(board, size):
    """
    :param board: initial nqueens configuration
    :param size: population size
    :return: a list of size random nqueens configurations
    """
    population = [board]
    for individual in range(0, size):
        for column, row in enumerate(board):  # For each column, place the queen in a random row
            board[column] = random.randint(0, len(board) - 1)
        population.append(board.copy())

    return population


def genetic_algorithm(board):
    """
    Implement this yourself.
    :param board: initial nqueens configuration
    """
    population_size = 20

    optimum = (len(board) - 1) * len(board) / 2
    generation = 0

    population = make_population(board, population_size)
    weights, best_individual = calculate_weights(population)

    while evaluate_state(best_individual) != optimum:

        generation += 1
        population2 = []
        print(f"gen: {generation} fitness: {evaluate_state(best_individual)}")

        for i in range(1, population_size):
            parent1, parent2 = random.choices(population, weights, k=2)
            child = reproduce(parent1, parent2)
            population2.append(mutate(child))

        population = population2
        weights, best_individual = calculate_weights(population)

    if evaluate_state(best_individual) == optimum:
        print('Solved puzzle!')

    print('Final state is:')
    print_board(best_individual)

def stochastic(board):
    """
    Implement this yourself.
    :param board:
    :return:
    """
    i = 0
    optimum = (len(board) - 1) * len(board) / 2
    neighbour = board.copy()
    selected_neighbour = board.copy()
    while evaluate_state(selected_neighbour) != optimum:
        better_neighbour = []
        neighbour_weights = []
        i += 1
        print('iteration ' + str(i) + ': evaluation = ' + str(evaluate_state(selected_neighbour)))
        if i > 1000:  # Give up after 1000 tries.
            break
        neighbour = board.copy()
        for column, row in enumerate(neighbour):  # For each column
            for qrow in range(0, len(board)):
                if row == qrow:
                    continue
                neighbour[column] = qrow
                if evaluate_state(neighbour) >= evaluate_state(selected_neighbour):
                    better_neighbour.append(neighbour.copy())
                    neighbour_weights.append(evaluate_state(neighbour) - evaluate_state(selected_neighbour) + 1)
                neighbour = board.copy()
        if len(better_neighbour) > 0:
            selected_neighbour = random.choices(better_neighbour, weights=neighbour_weights)[0]
        board = selected_neighbour.copy()


    if evaluate_state(selected_neighbour) == optimum:
        print('Solved puzzle!')

    print('Final state is:')
    print_board(board)

def main():
    """
    Main function that will parse input and call the appropriate algorithm. You do not need to understand everything
    here!
    """

    try:
        if len(sys.argv) != 2:
            raise ValueError

        n_queens = int(sys.argv[1])
        if n_queens < 1 or n_queens > MAXQ:
            raise ValueError

    except ValueError:
        print('Usage: python n_queens.py NUMBER')
        return False

    print('Which algorithm to use?')
    algorithm = input('1: random, 2: hill-climbing, 3: simulated annealing, 4: genetic algorithm, 5: stochastic\n')

    try:
        algorithm = int(algorithm)

        if algorithm not in range(1, 6):
            raise ValueError

    except ValueError:
        print('Please input a number in the given range!')
        return False

    board = init_board(n_queens)
    print('Initial board: \n')
    print_board(board)

    if algorithm == 1:
        random_search(board)
    if algorithm == 2:
        hill_climbing(board)
    if algorithm == 3:
        simulated_annealing(board)
    if algorithm == 4:
        genetic_algorithm(board)
    if algorithm == 5:
        stochastic(board)


# This line is the starting point of the program.
if __name__ == "__main__":
    main()

