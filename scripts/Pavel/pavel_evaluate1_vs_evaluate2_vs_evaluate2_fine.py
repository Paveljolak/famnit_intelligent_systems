# Two different evals

# evaluate1: 
# The first evaluate that we had.

# evaluate2:
# Evaluate 2 is better in this case.

# evalute2_fine - Fine grained evaluate function

import gymnasium as gym
from famnit_gym.envs import mill
from famnit_gym.wrappers.mill import DelayMove
import random

def minimax(model, depth, alpha, beta, maximizing_player):
    if (depth==0 or model.game_over()):
        eval_p1 = evaluate2_fine(model, 1)
        eval_p2 = evaluate2(model, 2)
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
        if depth==4:
            print(f'best for kurva1: {best_moves}')
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
            if depth==4:
                print(f'best for kurva2: {best_moves}')
        return min_eval, random.choice(best_moves) if best_moves else None
    
def evaluate1(model, player):
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
            1000 * count_morrises(state, player) +
            10 * blocked_pieces(state, player) +
            8 * model.count_pieces(player) +
            10000 * win(model, player)
        )
    else:
        evaluation = (
            10 * two_piece_configuration(state, player) +
            1190 * win(model, player)
        )

    return evaluation

def evaluate2(model, player):
    state = model.get_state()
    opponent = 2 if player == 1 else 1
    phase = model.get_phase(player)

    my_pieces = model.count_pieces(player)
    opp_pieces = model.count_pieces(opponent)
    my_morrises = count_morrises(state, player)
    opp_morrises = count_morrises(state, opponent)
    my_two_config = two_piece_configuration(state, player)
    opp_two_config = two_piece_configuration(state, opponent)
    blocked_opp = blocked_pieces(state, opponent)
    won = win(model, player)

    if phase == 'placing':
        # Focus on mills and blocking opponent
        evaluation = (
            50 * my_morrises - 40 * opp_morrises + 
            20 * my_two_config - 10 * opp_two_config +
            15 * blocked_opp + 
            10 * (my_pieces - opp_pieces)
        )

    elif phase == 'moving':
        # Piece advantage matters more, blocking opponent is stronger
        evaluation = (
            100 * my_morrises - 80 * opp_morrises +
            40 * my_two_config - 20 * opp_two_config +
            50 * blocked_opp +
            100 * (my_pieces - opp_pieces) +
            2000 * won
        )

    elif phase == 'flying':
        # Aggressive: maximize mills, piece advantage, immediate wins
        evaluation = (
            120 * (my_morrises - opp_morrises) +
            60 * my_two_config - 40 * opp_two_config +
            150 * (my_pieces - opp_pieces) +
            3000 * won
        )
    else:
        evaluation = 0

    return evaluation


def evaluate2_fine(model, player):
    state = model.get_state()
    opponent = 2 if player == 1 else 1
    phase = model.get_phase(player)

    my_pieces = model.count_pieces(player)
    opp_pieces = model.count_pieces(opponent)
    my_morrises = count_morrises(state, player)
    opp_morrises = count_morrises(state, opponent)
    my_two_config = two_piece_configuration(state, player)
    opp_two_config = two_piece_configuration(state, opponent)
    blocked_opp = blocked_pieces(state, opponent)
    blocked_me = blocked_pieces(state, player)
    mobility = len(model.legal_moves(player))  # add mobility as tie-breaker
    opp_mobility = len(model.legal_moves(opponent))
    won = win(model, player)

    if phase == 'placing':
        evaluation = (
            50.5 * my_morrises - 40.3 * opp_morrises +
            20.2 * my_two_config - 10.1 * opp_two_config +
            15.5 * blocked_opp +
            10.1 * (my_pieces - opp_pieces) +
            0.5 * mobility - 0.3 * opp_mobility
        )

    elif phase == 'moving':
        evaluation = (
            100.7 * my_morrises - 80.4 * opp_morrises +
            40.3 * my_two_config - 20.2 * opp_two_config +
            50.5 * blocked_opp - 10.2 * blocked_me +
            100.3 * (my_pieces - opp_pieces) +
            5.5 * mobility - 3.3 * opp_mobility +
            2000.1 * won
        )

    elif phase == 'flying':
        evaluation = (
            120.6 * (my_morrises - opp_morrises) +
            60.3 * my_two_config - 40.2 * opp_two_config +
            150.4 * (my_pieces - opp_pieces) +
            8.5 * mobility - 5.1 * opp_mobility +
            3000.5 * won
        )

    else:
        evaluation = 0

    return evaluation



def evaluate3(model, player):
    state = model.get_state()
    opponent = 2 if player == 1 else 1
    phase = model.get_phase(player)

    my_pieces = model.count_pieces(player)
    opp_pieces = model.count_pieces(opponent)
    my_morrises = count_morrises(state, player)
    opp_morrises = count_morrises(state, opponent)
    my_two_config = two_piece_configuration(state, player)
    opp_two_config = two_piece_configuration(state, opponent)
    blocked_opp = blocked_pieces(state, opponent)
    blocked_me = blocked_pieces(state, player)
    won = win(model, player)

    if phase == 'placing':
        # Aggressively build mills, block opponent, set up two-piece configs
        evaluation = (
            60 * my_morrises - 50 * opp_morrises +
            25 * my_two_config - 15 * opp_two_config +
            20 * blocked_opp +
            10 * (my_pieces - opp_pieces)
        )

    elif phase == 'moving':
        # Mobility, piece advantage, blocking, mill formation, win potential
        evaluation = (
            120 * my_morrises - 100 * opp_morrises +
            50 * my_two_config - 25 * opp_two_config +
            60 * blocked_opp - 20 * blocked_me +
            150 * (my_pieces - opp_pieces) +
            2500 * won
        )

    elif phase == 'flying':
        # Maximize immediate win chances and piece advantage
        evaluation = (
            200 * (my_morrises - opp_morrises) +
            80 * my_two_config - 50 * opp_two_config +
            200 * (my_pieces - opp_pieces) +
            5000 * won
        )
    else:
        evaluation = 0

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