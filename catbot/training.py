import random
from typing import Dict, List, Tuple
import numpy as np
from utility import play_q_table
from cat_env import make_env

#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################
import csv

# Hyperparameters for Q-Learning
ALPHA = 0.1  # Learning Rate (Will be decayed)
GAMMA = 0.99  # Discount Factor
EPSILON = 1.0  # Initial Exploration Rate
EPSILON_DECAY = 0.9995  # Decay rate per episode
MIN_EPSILON = 0.01  # Minimum Epsilon value
ALPHA_DECAY = 0.9999  # Decay rate for Learning Rate (Optimization)
MIN_ALPHA = 0.001  # Minimum Alpha value

# Global/Tracking Variables
current_epsilon = EPSILON
current_alpha = ALPHA


#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

# TODO: Remove Tuple and List return from final submission
def train_bot(cat_name, render: int = -1) -> Tuple[Dict[int, np.ndarray], List[int]]:
    # TODO: Override instructions and add global declaration
    global current_epsilon, current_alpha

    env = make_env(cat_type=cat_name)

    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000  # Training is capped at 5000 episodes for this project

    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################

    # Reset tracking variables for this run
    current_epsilon = EPSILON
    current_alpha = ALPHA
    steps_per_episode = []

    episode_interval_data = {}
    episode_interval = 500

    success_count = 0
    explorations_per_episode = []
    exploitations_per_episode = []

    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################

    for ep in range(1, episodes + 1):
        #############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                      #
        #############################################################################
        # Hint: These are the general steps you must implement for each episode.    #
        # 1. Reset the environment to start a new episode.                          #
        # 2. Decide whether to explore or exploit.                                  #
        # 3. Take the action and observe the next state.                            #
        # 4. Since this environment doesn't give rewards, compute reward manually   #
        # 5. Update the Q-table accordingly based on agent's rewards.               #
        #############################################################################

        # 1. Reset environment
        obs, info = env.reset()
        done = False
        steps = 0
        exploration_count = 0
        exploitation_count = 0

        max_training_steps = 250

        while not done:
            # 2. Decide whether to explore or exploit
            if random.random() < current_epsilon:
                action = env.action_space.sample()  # Explore
                exploration_count += 1
            else:
                # Exploit: Randomly break ties
                max_q = np.max(q_table[obs])
                best_actions = np.where(q_table[obs] == max_q)[0]
                action = random.choice(best_actions)
                exploitation_count += 1

            # 3. Take action and observe
            new_obs, _, terminated, truncated, info = env.step(action)
            steps += 1

            # 4. Compute reward manually
            reward = 0.0
            if terminated:
                reward = 100.0  # Cat caught
            elif env.cat.current_distance < env.cat.prev_distance:
                reward = 1.0  # Got closer
            elif env.cat.current_distance > env.cat.prev_distance:
                reward = -5.0  # Moved farther
            else:
                reward = -0.5  # Stalled

            # 5. Update Q-table (using the dictionary)
            old_value = q_table[obs][action]
            next_max = np.max(q_table[new_obs])

            # Q-Learning Formula
            new_value = (1 - current_alpha) * old_value + current_alpha * (reward + GAMMA * next_max)
            q_table[obs][action] = new_value

            obs = new_obs
            done = terminated or truncated

            if steps >= max_training_steps:
                done = True

            if terminated:
                success_count += 1

        # --- End of while loop ---

        # Record data
        steps_per_episode.append(steps)
        explorations_per_episode.append(exploration_count)
        exploitations_per_episode.append(exploitation_count)
        if ep % episode_interval == 0:
            episode_interval_stats = []

            print('------------------')
            print(f'Episode {ep}')
            # Record success count and rate
            success_rate = (success_count / episode_interval) * 100
            print(f'Success: {success_count} | {success_rate}')
            episode_interval_stats.append(success_count)
            episode_interval_stats.append(success_rate)
            
            # Record average no. of steps 
            average_steps = np.mean(steps_per_episode)
            print(f'Average No. of Steps: {average_steps}')
            episode_interval_stats.append(average_steps)

            # Record average exploration rate vs exploitation rate
            average_exploration_count = np.mean(explorations_per_episode)
            average_exploration_rate = round((average_exploration_count.item() / average_steps) * 100, 2)
            episode_interval_stats.append(average_exploration_count)
            episode_interval_stats.append(average_exploration_rate)

            average_exploitation_count = np.mean(exploitations_per_episode)
            average_exploitation_rate = round((average_exploitation_count.item() / average_steps) * 100, 2)
            episode_interval_stats.append(average_exploitation_count)
            episode_interval_stats.append(average_exploitation_rate)

            print(f'Exploration Average: {average_exploration_count} ({average_exploration_rate}%) | Exploitation Average: {average_exploitation_count} ({average_exploitation_rate}%)')
            
            # Record final epsilon decay at this episode interval
            print(f'{current_epsilon}')
            episode_interval_stats.append(round(current_epsilon, 3))
            
            episode_interval_data[ep] = episode_interval_stats

            # Reset env after the interval to get a new set of values
            success_count = 0
            steps_per_episode = []
            explorations_per_episode = []
            exploitations_per_episode = []
        
        # Decay hyperparameters
        current_epsilon = max(MIN_EPSILON, current_epsilon * EPSILON_DECAY)
        current_alpha = max(MIN_ALPHA, current_alpha * ALPHA_DECAY)

        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02,
                         window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)
    
    #print(episode_interval_data)
    
    # TODO: Remove steps_per_episode from final submission
    return q_table, steps_per_episode, episode_interval_data