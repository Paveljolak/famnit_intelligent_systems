    # The transition model includes the following public methods:
    # * clone()
    #   Creates a deep copy of the object.
    #
    # * get_state()
    #   Returns the board position as a list [0 - 23], containing values:
    #   0 (empty), 1 (player 1), 2 (player 2).
    #   Note: Board position 1 has index 0 in the list, etc.
    #
    # * get_phase(player)
    #   Returns the phase of the given player (placing, moving, flying).
    #
    # * count_pieces(player)
    #   Returns number of pieces on the board belonging to the given player.
    #
    # * legal_moves(player)
    #   Returns the list of legal moves for the given player.
    #
    # * make_move(player, move)
    #   Changes the state as if the given player made the given move.
    #   Note: The correctnes of the move is not checked for performance
    #         reasons. The user should only make moves from the list of
    #         of legal moves. Player's turn is also not checked. The same
    #         player can be simulated as making multiple consecutive moves.
    #
    # * game_over()
    #   Return True if one of the player has lost the game.
    #
    # Printing the transition model prints the board state in ASCII.

import gymnasium as gym
from famnit_gym.envs import mill
from famnit_gym.wrappers.mill import DelayMove
import random

def minimax(model, depth, alpha, beta, maximizing_player):
    if (depth==0 or model.game_over()):
        eval_p1 = evaluate(model, 1)
        eval_p2 = evaluate(model, 2)
        return eval_p1-eval_p2, None
    if maximizing_player:
        max_eval = float('-inf')
        best_moves = []
        for move in model.legal_moves(player=1):
            model_clone = model.clone()
            model_clone.make_move(player=1, move=move)
            evaluation, _ = minimax(model_clone, depth-1, alpha, beta, False)
            if evaluation > max_eval:
                max_eval = evaluation
                best_moves = [move]
            elif evaluation == max_eval:
                best_moves.append(move)
            alpha = max(alpha, max_eval )
            if beta <= alpha:
                break
        return max_eval, random.choice(best_moves) if best_moves else None
    else:
        min_eval = float('inf')
        best_moves = []
        for move in model.legal_moves(player=2):
            model_clone = model.clone()
            model_clone.make_move(player=2, move=move)
            evaluation, _ = minimax(model_clone, depth-1, alpha, beta, True)
            if evaluation < min_eval:
                min_eval = evaluation
                best_moves = [move]
            elif evaluation == min_eval:
                best_moves.append(move)
            beta = min(beta, min_eval)
            if beta <= alpha:
                break
        return min_eval, random.choice(best_moves) if best_moves else None
    
def evaluate(model, player):
    state = model.get_state()
    phase = model.get_phase(player)
    if(phase == 'placing'):
        evaluation = (
            26 * count_morrises(state, player) +
            blocked_pieces(state, player) +
            6 * model.count_pieces(player) +
            12 * two_piece_configuration(state, player)
        )
    elif(phase == 'moving'):
        evaluation = (
            43 * count_morrises(state, player) +
            10 * blocked_pieces(state, player) +
            8 * model.count_pieces(player) +
            1086 * win(model, player)
        )
    else:
        evaluation = (
            10 * two_piece_configuration(state, player) +
            1190 * win(model, player)
        )

    return evaluation


def count_morrises(state, player):
    count = 0
    for line in morris_lines:
        if state[line[0]] == state[line[1]] == state[line[2]] == player:
            count += 1
    return count

def blocked_pieces(state, player):
    opponent = 2 if player == 1 else 1
    count = 0
    for i in range(24):
        for pos in adjecent_positions[i]:
            if state[i] == opponent and pos == player:
                count += 1
    return count

def two_piece_configuration(state, player):
    count = 0
    for line in morris_lines:
        if((state[line[0]] == state[line[1]] == player and state[line[2]] == 0) or
           (state[line[1]] == state[line[2]] == player and state[line[0]] == 0) or
           (state[line[0]] == state[line[2]] == player and state[line[1]] == 0)):
            count += 1
    return count

def win(model, player):
    opponent = 2 if player == 1 else 1

    if model.count_pieces(opponent) < 3 or model.legal_moves(opponent) == []:
        return 1
        
    return 0

delay = False

env = mill.env(render_mode="human")
if delay:
    env = DelayMove(env, time_limit=100)
env.reset()

morris_lines = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23], #horizontal
    [0, 9, 21], [3, 10, 18], [6, 11, 15], [1, 4, 7], [16, 19, 22], [8, 12, 17], [5, 13, 20], [2, 14, 23], #vertical
    [0, 3, 6], [2, 5, 8], [17, 20, 23], [15, 18, 21] #diagonal
]

adjecent_positions = [
    [1, 3, 9], [0, 2, 4], [1, 5, 14], [0, 4, 6, 10], [1, 3, 5, 7], [2, 4, 8, 13], [3, 7, 11], [4, 6, 8], [5, 7, 12], [0, 10, 21], [3, 9, 11, 18], [6, 10, 15],
    [8, 13, 17], [5, 12, 14, 20], [2, 13, 23], [11, 16, 18], [15, 17, 19], [12, 16, 20], [10, 15, 19, 21], [16, 18, 20, 22], [13, 17, 19, 23], [9, 18, 22], [19, 21, 23], [14, 20, 22]
]

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination:
        print(f"{agent} lost the game!")
        break

    if truncation:
        print("The game was too long!")
        break

    if delay:
        model = mill.transition_model(env.env) #Ako DelayMove ukljucen, onda ovaj model, ako ne onda drugi
    else:
        model = mill.transition_model(env)
    
    eval, move = minimax(model, 4, float('-inf'), float('inf'), True if agent=="player_1" else False)

    #print(model)
    print(eval, move)

    env.step(move)

# Close the environment.
env.close()