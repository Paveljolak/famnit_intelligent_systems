# SUMMARY 
# _____________________________________________________________________________________________________________

# Measuring the time minimax needs for each depth

# _____________________________________________________________________________________________________________


# IMPORTS 
# _____________________________________________________________________________________________________________

import gymnasium as gym
from famnit_gym.envs import mill
from famnit_gym.wrappers.mill import DelayMove
import random
import time
import matplotlib.pyplot as plt

# _____________________________________________________________________________________________________________


# BOARD SETUP
# _____________________________________________________________________________________________________________

# Different Possible Mills
morris_lines = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23], #horizontal
    [0, 9, 21], [3, 10, 18], [6, 11, 15], [1, 4, 7], [16, 19, 22], [8, 12, 17], [5, 13, 20], [2, 14, 23], #vertical
    [0, 3, 6], [2, 5, 8], [17, 20, 23], [15, 18, 21] #diagonal
]

# Each index represents the point on that position, each array on that index is the positions of the adjacent points to that point
adjecent_positions = [
    [1, 3, 9], [0, 2, 4], [1, 5, 14], [0, 4, 6, 10], [1, 3, 5, 7], [2, 4, 8, 13], [3, 7, 11], [4, 6, 8], [5, 7, 12], [0, 10, 21], [3, 9, 11, 18], [6, 10, 15],
    [8, 13, 17], [5, 12, 14, 20], [2, 13, 23], [11, 16, 18], [15, 17, 19], [12, 16, 20], [10, 15, 19, 21], [16, 18, 20, 22], [13, 17, 19, 23], [9, 18, 22], [19, 21, 23], [14, 20, 22]
]

# _____________________________________________________________________________________________________________


# MINIMAX
# _____________________________________________________________________________________________________________

def minimax(model, depth, alpha, beta, maximizing_player):
    if (depth==0 or model.game_over()):
        eval_p1 = evaluate1(model, 1)
        eval_p2 = evaluate1(model, 2)
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
 
# _____________________________________________________________________________________________________________



# EVALUATION HELPER FUNCTIONS
# _____________________________________________________________________________________________________________

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

# _____________________________________________________________________________________________________________


# EVALUATION FUNCTION - evaluate1
# _____________________________________________________________________________________________________________

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




# Space to plug a different evaluation function to get it tested whether it will change something on the 
# actual minimax time.



# _____________________________________________________________________________________________________________



# ENVIRONMENT
# _____________________________________________________________________________________________________________

env = mill.env(render_mode="human")
env.reset()
model = mill.transition_model(env)

# _____________________________________________________________________________________________________________


# TIME ANALYSIS WITH AVERAGE OVER 10 RUNS
# _____________________________________________________________________________________________________________

depths = [1, 2, 3, 4, 5, 6] # Try 7 as well, but it will take some time
num_runs = 10
all_times = []
avg_times = []

for d in depths:
    run_times = []
    for i in range(num_runs):
        start = time.perf_counter()
        minimax(model.clone(), d, float('-inf'), float('inf'), True)
        end = time.perf_counter()
        run_times.append(end - start)
        print(f"Depth {d}, Run {i+1}: {end - start:.4f} seconds")
    all_times.append(run_times)
    avg_time = sum(run_times) / num_runs
    avg_times.append(avg_time)
    print(f"Depth {d}: Average time over {num_runs} runs = {avg_time:.4f} seconds\n")

# PLOT AVERAGE TIMES
plt.plot(depths, avg_times, marker='o')
plt.xlabel("Depth")
plt.ylabel("Average Time (seconds)")
plt.title(f"Average Minimax Search Time vs Depth over {num_runs} runs")
plt.grid(True)
plt.show()

# _____________________________________________________________________________________________________________


env.close()
