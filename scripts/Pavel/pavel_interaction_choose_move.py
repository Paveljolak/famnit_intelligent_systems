# Interaction with choosing which piece to move against random AI

import gymnasium as gym
import random
from famnit_gym.envs import mill
from famnit_gym.wrappers.mill import UserInteraction
import sys

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

    # --- Human player ---
    if agent == "player_1":
        env.clear_markings()
        done_interacting = False

        while not done_interacting:
            event = env.interact()

            # Quit the game
            if event["type"] == "quit":
                print("Game quit by window close.")
                env.close()
                sys.exit(0) # Actually close the window 

            # Quit the game with ESC key
            elif event["type"] == "key_press" and event["key"] == "escape":
                print("Game quit by ESC key.")
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


                # PLACING PHASE
                # --------------------------
                if any(move[0] == 0 and move[1] == pos for move in legal_moves):
                    for [src, dst, cap] in legal_moves:
                        if dst == pos:
                            move = [src, dst, cap]
                            done_interacting = True
                            break
                # --------------------------

                # MOVING/FLYING PHASE
                # --------------------------
                else:
                    if selected_piece is None and observation[pos - 1] == 1:
                        # Select a piece to move
                        selected_piece = pos
                        env.clear_markings()
                        for src, dst, _ in legal_moves:
                            if src == selected_piece:
                                env.mark_position(dst, (255, 255, 0, 128))  # yellow

                    elif selected_piece is not None:
                        if observation[pos - 1] == 1:
                            # Reselect a piece (Clicked on another piece)
                            selected_piece = pos
                            env.clear_markings()
                            for src, dst, _ in legal_moves:
                                if src == selected_piece:
                                    env.mark_position(dst, (255, 255, 0, 128))  # yellow
                        else:
                            # Click a destination square
                            for src, dst, cap in legal_moves:
                                if src == selected_piece and dst == pos:
                                    move = [src, dst, cap]
                                    selected_piece = None
                                    done_interacting = True
                                    break

        env.clear_markings()

    # AI PLAYER - RANDOM
    else:
        legal_moves = info["legal_moves"]
        move = random.choice(legal_moves.tolist()) if len(legal_moves) > 0 else None

    env.step(move)

env.close()
