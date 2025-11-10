# Interaction with choosing which piece to move also which piece to capture against Luka's Minimax

import gymnasium as gym
import random
from famnit_gym.envs import mill
from famnit_gym.wrappers.mill import UserInteraction
import sys


# LUKA'S MINIMAX FUNCTION
# -------------------------------------------------------------
def minimax(model, depth, alpha, beta, maximizing_player):
    if (depth==0 or model.game_over()):
        eval = evaluate(model)
        return eval, None
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
        if depth == 3:
            print(max_eval, best_moves)
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
        if depth == 3:
            print(min_eval, best_moves)
        return min_eval, random.choice(best_moves) if best_moves else None
# -------------------------------------------------------------


# LUKA'S EVAL FUNCTION
# We can improve it and it will be even more difficult.
# This can be difficulty 1. 
# -------------------------------------------------------------
def evaluate(model):
    evaluation = model.count_pieces(1) - model.count_pieces(2)
    return evaluation
# -------------------------------------------------------------

env = mill.env(render_mode='human')
env = UserInteraction(env)
env.reset()

selected_piece = None

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination:
        print(f"{agent} lost the game!")
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
        eval, move = minimax(model, 3, float('-inf'), float('inf'), True if agent=="player_1" else False)
    env.step(move)
    # -------------------------------------------------------------

env.close()
