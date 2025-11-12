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

    #print(eval, move)

    env.step(move)

# Close the environment.
env.close()