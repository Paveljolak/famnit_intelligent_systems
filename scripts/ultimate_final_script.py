# Interaction with choosing which piece to move also which piece to capture against Luka's Minimax

import gymnasium as gym
import random
import math
from famnit_gym.envs import mill
from famnit_gym.wrappers.mill import UserInteraction
import sys

difficulty = 0
mode = ""

while mode != 1 and mode != 2:
    print("Choose game mode: \n 1. AI vs AI \n 2. Player vs AI \n Insert 1 or 2: ")
    mode = int(input().strip())


# LUKA'S MINIMAX FUNCTION
# -------------------------------------------------------------
morris_lines = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23], #horizontal
    [0, 9, 21], [3, 10, 18], [6, 11, 15], [1, 4, 7], [16, 19, 22], [8, 12, 17], [5, 13, 20], [2, 14, 23], #vertical
    [0, 3, 6], [2, 5, 8], [17, 20, 23], [15, 18, 21] #diagonal
]

adjecent_positions = [
    [1, 3, 9], [0, 2, 4], [1, 5, 14], [0, 4, 6, 10], [1, 3, 5, 7], [2, 4, 8, 13], [3, 7, 11], [4, 6, 8], [5, 7, 12], [0, 10, 21], [3, 9, 11, 18], [6, 10, 15],
    [8, 13, 17], [5, 12, 14, 20], [2, 13, 23], [11, 16, 18], [15, 17, 19], [12, 16, 20], [10, 15, 19, 21], [16, 18, 20, 22], [13, 17, 19, 23], [9, 18, 22], [19, 21, 23], [14, 20, 22]
]

def minimax(model, depth, alpha, beta, maximizing_player):
    if model.game_over():
        if maximizing_player:
            eval = -10000
        else:
            eval= 10000
        return eval, None
    elif depth == 0:
        eval_p1 = evaluate(model, 1)
        eval_p2 = evaluate(model, 2)
        return eval_p1-eval_p2, None
    elif maximizing_player:
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

    if difficulty == 1:
        return model.count_pieces(player)

    if(phase == 'placing'):
        evaluation = (
            20 * count_morrises(state, player) +
            8 * blocked_pieces(state, player) +
            8 * model.count_pieces(player) +
            10 * two_piece_configuration(state, player)
        )
    elif(phase == 'moving'):
        evaluation = (
            40 * count_morrises(state, player) +
            10 * blocked_pieces(state, player) +
            8 * model.count_pieces(player)
        )
    else:
        evaluation = (
            15 * model.count_pieces(player) + 
            10 * two_piece_configuration(state, player)
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
# -------------------------------------------------------------
if mode == 2:
    while difficulty != 1 and difficulty != 2 and difficulty != 3:
        print("Choose game difficulty: \n 1. Beginner \n 2. Intermediate \n 3. Master \n Insert 1, 2 or 3: ")
        difficulty = int(input().strip())

    env = mill.env(render_mode='human')
    env = UserInteraction(env)
    env.reset()

    selected_piece = None

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination:
            print(f"{"Player 1" if agent == "player_1" else "Player 2"} lost the game!")
            break

        if truncation:
            print("The game is too long or user quit!")
            break

        move = None

        # HUMAN PLAYER
        # -------------------------------------------------------------
        if agent == "player_1":
            env.clear_markings()
            done_interacting = False
            selected_piece = None
            src, dst = None, None
            awaiting_capture = False
            chosen_capture = None

            while not done_interacting:
                event = env.interact()

                # QUIT 
                if event["type"] == "quit" or (event["type"] == "key_press" and event["key"] == "escape"):
                    print("Game quit by user.")
                    env.close()
                    sys.exit(0)

                elif event["type"] == "mouse_move":
                    pos = event["position"]
                    if pos != 0:
                        if observation[pos - 1] == 0:
                            env.set_selection_color((64, 192, 0, 128))  # green for empty
                        else:
                            env.set_selection_color((128, 128, 255, 255))  # blue for occupied

                elif event["type"] == "mouse_click":
                    pos = event["position"]
                    if pos == 0:
                        continue

                    legal_moves = info["legal_moves"]

                    # CAPTURE SELECTION 
                    # -------------------------------------------------------------
                    if awaiting_capture:
                        opponent_pieces = [m[2] for m in legal_moves if m[0] == src and m[1] == dst and m[2] > 0]
                        
                        # Highlight capturable pieces immediately
                        env.clear_markings()
                        for pos_idx in opponent_pieces:
                            env.mark_position(pos_idx, (255, 0, 0, 128))  # red

                        if pos in opponent_pieces:
                            chosen_capture = pos
                            move = [src, dst, chosen_capture]
                            done_interacting = True
                            awaiting_capture = False
                            env.clear_markings()
                            break

                    # PLACING / MOVING / FLYING
                    # ---------------------------------------------------------------------------------------------------
                    else:
                        # PLACING 
                        # -------------------------------------------------------------
                        if any(m[0] == 0 and m[1] == pos for m in legal_moves):
                            for s, d, c in legal_moves:
                                if d == pos:
                                    move = [s, d, c]
                                    if c > 0:
                                        src, dst = s, d
                                        awaiting_capture = True
                                        move = None
                                        # Highlight capturable pieces
                                        opponent_pieces = [m[2] for m in legal_moves if m[0] == src and m[1] == dst and m[2] > 0]
                                        env.clear_markings()
                                        for pos_idx in opponent_pieces:
                                            env.mark_position(pos_idx, (255, 0, 0, 128))
                                    else:
                                        done_interacting = True
                                    break
                        # -------------------------------------------------------------
                        
                        # MOVING / FLYING PHASE 
                        # -------------------------------------------------------------
                        else:
                            if selected_piece is None and observation[pos - 1] == 1:
                                selected_piece = pos
                                env.clear_markings()
                                for s, d, c in legal_moves:
                                    if s == selected_piece:
                                        env.mark_position(d, (255, 255, 0, 128))  # yellow

                            elif selected_piece is not None:
                                if observation[pos - 1] == 1:
                                    selected_piece = pos
                                    env.clear_markings()
                                    for s, d, c in legal_moves:
                                        if s == selected_piece:
                                            env.mark_position(d, (255, 255, 0, 128))  # yellow
                                else:
                                    for s, d, c in legal_moves:
                                        if s == selected_piece and d == pos:
                                            move = [s, d, c]
                                            src, dst = s, d
                                            selected_piece = None
                                            if c > 0:
                                                awaiting_capture = True
                                                move = None
                                                # Highlight capturable pieces
                                                opponent_pieces = [m[2] for m in legal_moves if m[0] == src and m[1] == dst and m[2] > 0]
                                                env.clear_markings()
                                                for pos_idx in opponent_pieces:
                                                    env.mark_position(pos_idx, (255, 0, 0, 128))
                                            else:
                                                done_interacting = True
                                            break
                        # -------------------------------------------------------------

            env.clear_markings()

        # AI PLAYER (LUKA'S MINIMAX)
        # -------------------------------------------------------------
        else:
            model = mill.transition_model(env.env) #Ako DelayMove ukljucen, onda ovaj model, ako ne onda drugi
            eval, move = minimax(model, math.ceil(difficulty*1.5), float('-inf'), float('inf'), True if agent=="player_1" else False)
        env.step(move)
        # -------------------------------------------------------------

elif mode == 1:
    env = mill.env(render_mode="human")
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination:
            print(f"{"Player 1" if agent == "player_1" else "Player 2"} lost the game!")
            break

        if truncation:
            print("The game was too long!")
            break

        model = mill.transition_model(env)
        
        eval, move = minimax(model, 5, float('-inf'), float('inf'), True if agent=="player_1" else False)

        env.step(move)

env.close()