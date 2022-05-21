import sys


def max_value(state):
    max = -100000000000

    if state == 1:
        return -1

    for move in range(1, 4):
        if state-move > 0:
            m = min_value(state-move)
            max = m if m > max else max

    return max


def min_value(state):
    min = 10000000000000

    if state == 1:
        return 1

    for move in range(1, 4):
        if state-move > 0:
            m = max_value(state-move)
            min = m if m < min else min

    return min


def negamax(state, turn, max_tt, min_tt):
    best_move = None
    value = 0

    if state == 1:
        if turn == 0:
            return -1, 0
        else:
            return 1, 0

    if turn == 0:  # MAX' turn
        max = -100000000000

        # for all moves, make the one with the best guaranteed value
        for move in range(1, 4):
            if state - move > 0:
                if min_tt.get(state-move) is not None:
                    m = min_tt[state-move]
                else:
                    m, min_move = negamax(state - move, 1, max_tt, min_tt)
                    min_tt[state - move] = m
                if m > max:
                    max = m
                    best_move = move
                    value = max

    else:
        min = 10000000000000

        # pick move with worst guaranteed value
        for move in range(1, 4):
            if state - move > 0:
                if max_tt.get(state-move) is not None:
                    m = max_tt[state - move]
                else:
                    m, max_move = negamax(state-move, 0, max_tt, min_tt)
                    max_tt[state-move] = m
                if m < min:
                    min = m
                    best_move = move
                    value = min

    return value, best_move


def minimax(state, turn, max_tt, min_tt):
    best_move = None

    if turn == 0:  # MAX' turn
        max = -100000000000

        # for all moves, make the one with the best guaranteed value
        for move in range(1, 4):
            if state - move > 0:
                m = min_value(state - move)
                if m > max:
                    max = m
                    best_move = move

    else:
        min = 10000000000000

        # pick move with worst guaranteed value
        for move in range(1, 4):
            if state - move > 0:
                m = max_value(state-move)
                if m < min:
                    min = m
                    best_move = move

    return 0, best_move


def play_nim(state):
    turn = 0
    max_tt = {}
    min_tt = {}

    while state != 1:
        value, move = negamax(state, turn, max_tt, min_tt)
        print(str(state) + ": " + ("MAX" if not turn else "MIN") + " takes " + str(move))

        state -= move
        turn = 1 - turn

    print("1: " + ("MAX" if not turn else "MIN") + " looses")


def main():
    """
    Main function that will parse input and call the appropriate algorithm. You do not need to understand everything
    here!
    """

    try:
        if len(sys.argv) != 2:
            raise ValueError

        state = int(sys.argv[1])
        if state < 1 or state > 100:
            raise ValueError

        play_nim(state)

    except ValueError:
        print('Usage: python nim.py NUMBER')
        return False


if __name__ == '__main__':
    main()
