from tqdm import tqdm
from Experience import Experience

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class GridWorld(object):
    def __init__(self, size, obstacles, treasure, snakepit):
        """
        Initializes the gridworld.
        :param size: Size of the gridworld
        :param obstacles: List of the coordinates of obstacles in the gridworld
        :param treasure: Coordinates of the treasure in the gridworld
        :param snakepit: Coordinates of the snakepit in the gridworld
        """
        self.size = size
        self.snakepit = snakepit
        self.obstacles = obstacles
        self.treasure = treasure
        self.actions = {
            0: 'UP',
            1: 'RIGHT',
            2: 'DOWN',
            3: 'LEFT'
        }
        self.rewards = self.create_reward_grid()
        self.transactions = self.create_transaction_grid()
        self.policy = self.initialize_policy_grid()

    def create_reward_grid(self):
        """
        Creates a grid of rewards based on the size of the gridworld.
        :return: matrix of rewards for each state
        """
        # initialize reward grid to -1 everywhere
        rewards = np.full((self.size, self.size), -1)

        # set reward to -50 for snakepit states
        rewards[self.snakepit[0], self.snakepit[1]] = -50

        # set reward to 50 for treasure states
        rewards[self.treasure[0], self.treasure[1]] = 50

        # set reward to 0 for obstacle states
        for obstacle in self.obstacles:
            rewards[obstacle[0], obstacle[1]] = 0
        return rewards

    def initialize_policy_grid(self):
        """
        Creates a grid of random policies based on the size of the gridworld.
        :return: matrix of policies for each state
        """
        # initialize policy grid to 0 everywhere
        policy = np.random.randint(min(self.actions), max(self.actions), (self.size, self.size))

        # set policy to -1 for snakepit and treasure states
        policy[self.snakepit[0], self.snakepit[1]] = -1
        policy[self.treasure[0], self.treasure[1]] = -1

        # set policy to -1 for obstacle states
        for obstacle in self.obstacles:
            policy[obstacle[0], obstacle[1]] = -1
        return policy

    def create_transaction_grid(self):
        """
        Creates a grid of transactions for each action.
        :return: 3D matrix of transactions for each state and action
        """
        transaction_matrix = [[[(i, j) for _ in range(len(self.actions))] for j in range(self.size)]for i in range(self.size)]

        for action in self.actions:
            # UP transaction
            if self.actions[action] == 'UP':
                for i in range(self.size):
                    for j in range(self.size):
                        # if you are in the top row or under the obstacles, then set the transaction to the same state
                        if i == 0 or (i, j) in self.obstacles or (i - 1, j) in self.obstacles or (i, j) == self.snakepit or (i, j) == self.treasure:
                            transaction_matrix[i][j][action] = (i, j)
                        else:
                            # otherwise, set the transaction to the state above
                            transaction_matrix[i][j][action] = (i - 1, j)

            # DOWN transaction
            elif self.actions[action] == 'DOWN':
                for i in range(self.size):
                    for j in range(self.size):
                        # if you are in the bottom row or over the obstacles, then set the transaction to the same state
                        if i == self.size-1 or (i, j) in self.obstacles or (i + 1, j) in self.obstacles or (i, j) == self.snakepit or (i, j) == self.treasure:
                            transaction_matrix[i][j][action] = (i, j)
                        else:
                            # otherwise, set the transaction to the state below
                            transaction_matrix[i][j][action] = (i + 1, j)

            # LEFT transaction
            elif self.actions[action] == 'LEFT':
                for i in range(self.size):
                    for j in range(self.size):
                        # if you are in the first column or on the right of obstacles, then go to the same state
                        if j == 0 or (i, j) in self.obstacles or (i, j - 1) in self.obstacles or (i, j) == self.snakepit or (i, j) == self.treasure:
                            transaction_matrix[i][j][action] = (i, j)
                        else:
                            # otherwise, go to the state to the left
                            transaction_matrix[i][j][action] = (i, j - 1)

            # RIGHT transaction
            elif self.actions[action] == 'RIGHT':
                for i in range(self.size):
                    for j in range(self.size):
                        # if you are in the last column or on the left of obstacles, then go to the same state
                        if j == self.size-1 or (i, j) in self.obstacles or (i, j + 1) in self.obstacles or (i, j) == self.snakepit or (i, j) == self.treasure:
                            transaction_matrix[i][j][action] = (i, j)
                        else:
                            # otherwise, go to the state to the right
                            transaction_matrix[i][j][action] = (i, j + 1)
        return transaction_matrix

    def get_random_state(self):
        """
        Returns a random state from the grid that is different from the obstacles or absorbing states.
        :return: random state
        """
        state = (np.random.randint(0, self.size), np.random.randint(0, self.size))
        while state == self.treasure or state == self.snakepit or state in self.obstacles:
            state = (np.random.randint(0, self.size), np.random.randint(0, self.size))
        return state

    def generate_episode(self, state):
        """
        Generates a random episode using the equiprobable policy.
        :param state: starting state
        :return: list of states in the episode
        """
        episode = []
        while not (state == self.treasure or state == self.snakepit):
            old_state = state
            action = np.random.choice(list(self.actions.keys()))
            state = self.transactions[state[0]][state[1]][action]
            reward = self.rewards[state[0], state[1]]

            episode.append(Experience(old_state, action, reward, state))
        return episode

    def monte_carlo_equiprobable_policy_evaluation(self, num_episodes, discount_factor=1.0, every_visit=False):
        """
        Evaluates the value function using the Monte Carlo method.
        :param num_episodes: Number of episodes to use for evaluation
        :param discount_factor: Value of discount factor
        :param every_visit: Boolean indicating whether to use first visit MC or every visit MC
        :return: Value function matrix
        """

        # initialize the value function to zero
        v_values = np.zeros((self.size, self.size))

        # initialize and array of rewards for each state
        returns = {(i, j): list() for i in range(self.size) for j in range(self.size)}

        for _ in tqdm(range(num_episodes), desc='Evaluating value function using Monte Carlo method', unit=' episodes'):
            # choose an initial state different from the absorbing states or the obstacles
            start_state = self.get_random_state()
            # generate an episode
            episode = self.generate_episode(start_state)

            # for every visit MC initialize a dictionary of states in the episode and the list of rewards to average
            if every_visit:
                rewards_in_episode = {step.state: list() for step in episode}

            # compute the returns for each state in the episode
            G = 0
            for i, step in enumerate(episode[::-1]):
                G = discount_factor * G + step.reward

                if every_visit:
                    # every visit MC add the reward to the list for the state and when reach the last, average the list
                    idx = (step.state[0], step.state[1])
                    rewards_in_episode[idx].append(G)
                    if step.state not in [x.state for x in episode][:len(episode) - (i + 1)]:
                        returns[idx].append(np.average(rewards_in_episode[idx]))
                        v_values[idx] = np.average(returns[idx])
                else:
                    # first visit MC evaluation average the returns only for the first time the state is visited
                    if step.state not in [x.state for x in episode][:len(episode) - (i + 1)]:
                        idx = (step.state[0], step.state[1])
                        returns[idx].append(G)
                        v_values[idx] = np.average(returns[idx])
        return v_values

    def greedification(self, q_values):
        """
        Greedification of the value-state function and compute the corresponding policy matrix.
        :param q_values: Value-state function matrix
        """
        # greedify the value-state function
        for i in range(self.size):
            for j in range(self.size):
                self.policy[i, j] = np.argmax(q_values[i, j])

    def sarsa_algorithm(self, num_episodes, epsilon, alpha,  discount_factor=1.0, return_stats=False):
        """
        Implementation of the SARSA algorithm.
        :param num_episodes: Number of episodes to use for training
        :param epsilon: Value of epsilon for the epsilon-greedy policy
        :param alpha: Learning rate
        :param discount_factor: Value of discount factor
        :param return_stats: Boolean indicating whether to return the stats of the algorithm
        :return: Return the optimal Q-value matrix, the optimal policy is stored in the attribute policy
        """
        # initialize the value-state function arbitrally
        q_values = np.random.random((self.size, self.size, len(self.actions)))

        # store the stats of the algorithm in a dictionary
        stats = {
            'episode_times': np.zeros(num_episodes),
            'episode_steps': np.zeros(num_episodes),
            'avg_td': np.zeros(num_episodes),
            'std_td': np.zeros(num_episodes)
        }

        for e in tqdm(range(num_episodes), desc='Compute optimal policy using SARSA algorithm', unit=' episodes'):
            stats['episode_times'][e] = time.time()

            # choose an initial state different from the absorbing states or the obstacles
            state = self.get_random_state()

            # compute the policy derived from the Q-values
            self.greedification(q_values)

            # choose an action based on the policy using epsilon-greedy
            if np.random.random() < 1-epsilon:
                action = self.policy[state]
            else:
                action = np.random.choice(list(self.actions.keys()))

            # store an array of temporal differencing
            td = []

            t = 1
            # for each step in the episode
            while not (state == self.treasure or state == self.snakepit):
                old_action = action
                old_state = state
                state = self.transactions[old_state[0]][old_state[1]][old_action]
                reward = self.rewards[state[0], state[1]]

                # choose an action based on the policy using epsilon-greedy
                if np.random.random() < 1 - epsilon:
                    action = self.policy[state]
                else:
                    action = np.random.choice(list(self.actions.keys()))

                # this learning rate guarantees convergence
                lr = alpha / t
                # update the Q-values
                td_t = reward + discount_factor * q_values[state[0], state[1], action] - q_values[old_state[0], old_state[1], old_action]
                td.append(td_t)
                q_values[old_state[0], old_state[1], old_action] += lr * td_t
                t += 1

            stats['avg_td'][e] = np.average(td)
            stats['std_td'][e] = np.std(td)
            stats['episode_times'][e] = time.time() - stats['episode_times'][e]
            stats['episode_steps'][e] = t

        # update optimal policy greedified from the Q-values
        self.greedification(q_values)
        return q_values if not return_stats else q_values, stats

    def q_learning_algorithm(self, num_episodes, epsilon, alpha,  discount_factor=1.0, return_stats=False):
        """
        Implementation of the Q-learning algorithm.
        :param num_episodes: Number of episodes to use for training
        :param epsilon: Value of epsilon for the epsilon-greedy policy
        :param alpha: Learning rate
        :param discount_factor: Value of discount factor
        :param return_stats: Boolean indicating whether to return the stats of the algorithm
        :return: Return the optimal Q-value matrix, the optimal policy is stored in the attribute policy
        """
        # initialize the value-state function arbitrally
        q_values = np.random.random((self.size, self.size, len(self.actions)))

        # store the stats of the algorithm in a dictionary
        stats = {
            'episode_times': np.zeros(num_episodes),
            'episode_steps': np.zeros(num_episodes),
            'avg_td': np.zeros(num_episodes),
            'std_td': np.zeros(num_episodes)
        }

        for e in tqdm(range(num_episodes), desc='Compute optimal policy using Q-LEARNING algorithm', unit=' episodes'):
            stats['episode_times'][e] = time.time()

            # choose an initial state different from the absorbing states or the obstacles
            state = self.get_random_state()

            # compute the policy derived from the Q-values
            self.greedification(q_values)

            # store an array of temporal differencing
            td = []

            t = 1
            # for each step in the episode
            while not (state == self.treasure or state == self.snakepit):

                # choose an action based on the policy using epsilon-greedy
                if np.random.random() < 1 - epsilon:
                    action = self.policy[state]
                else:
                    action = np.random.choice(list(self.actions.keys()))

                new_state = self.transactions[state[0]][state[1]][action]
                reward = self.rewards[new_state[0], new_state[1]]

                # this learning rate guarantees convergence
                lr = alpha / t
                # update the Q-values
                td_t = reward + discount_factor * np.max(q_values[new_state[0], new_state[1]]) - q_values[state[0], state[1], action]
                td.append(td_t)
                q_values[state[0], state[1], action] += lr * td_t

                t += 1
                state = new_state

            stats['avg_td'][e] = np.average(td)
            stats['std_td'][e] = np.std(td)
            stats['episode_times'][e] = time.time() - stats['episode_times'][e]
            stats['episode_steps'][e] = t

        # update optimal policy gredified from the Q-values
        self.greedification(q_values)
        return q_values if not return_stats else q_values, stats

    def compute_v_values(self, q_values):
        """
        Compute the value function matrix from the Q-values matrix.
        :param q_values: Q-values matrix
        :return: Value function matrix
        """
        v_values = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                # compute the value function for each state
                if (i,j) == self.treasure or (i,j) == self.snakepit or (i,j) in self.obstacles:
                    v_values[i, j] = 0  # the value of the absorbing states and obstacles is 0
                else:
                    v_values[i, j] = np.max(q_values[i, j])
        return v_values

    def plot_gridworld(self, title=""):
        """
        Plots the gridworld with the rewards in each state.
        :return: None
        """

        plt.figure(figsize=(8, 8), dpi=80)

        # This dictionary defines the colormap
        cdict = {'green': ((0.0, 0.0, 0.0),  # no red at 0
                           (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                           (1.0, 0.4, 0.4)),  # set to 0.8 so its not too bright at 1

                 'red': ((0.0, 0.6, 0.6),  # set to 0.8 so its not too bright at 0
                         (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                         (1.0, 0.0, 0.0)),  # no green at 1

                 'blue': ((0.0, 0.0, 0.0),  # no blue at 0
                          (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                          (1.0, 0.0, 0.0))  # no blue at 1
                 }

        # Create the colormap using the dictionary
        greenRedCMap = colors.LinearSegmentedColormap('GreenRedCMap', cdict)

        # change values in reward for visualization purposes
        r = self.rewards.copy()
        r[r[:, :] == -1] = -20

        plt.imshow(r, cmap=greenRedCMap, interpolation='nearest')
        plt.xticks(np.arange(self.size), labels=np.arange(self.size))
        plt.yticks(np.arange(self.size), labels=np.arange(self.size))

        # Loop over data dimensions and create text annotations.
        for i in range(self.size):
            for j in range(self.size):
                if (i,j) == self.treasure or (i,j) == self.snakepit:
                    text = f"ABSORB\n{self.rewards[i, j]:.0f}"
                    weight = 'bold'
                elif (i,j) in self.obstacles:
                    text = "WALL"
                    weight = 'bold'
                else:
                    text = f"{self.rewards[i, j]:.0f}"
                    weight = 'normal'
                plt.text(j, i, text, ha="center", va="center", color="black", weight=weight)
        plt.title(title)

        ax = plt.gca()
        # Minor ticks
        ax.set_xticks(np.arange(-.5, 9, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 9, 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
        plt.tight_layout()
        plt.show()

    def plot_value_function(self, v_values, plot_policy=False, title=""):
        """
        Plots the value function matrix.
        :param v_values: Value function matrix
        :param plot_policy: Boolean to plot the optimal policy
        :return: None
        """
        plt.figure(figsize=(8, 8), dpi=80)

        # This dictionary defines the colormap
        cdict = {'green': ((0.0, 0.0, 0.0),  # no red at 0
                           (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                           (1.0, 0.4, 0.4)),  # set to 0.8 so its not too bright at 1

                 'red': ((0.0, 0.6, 0.6),  # set to 0.8 so its not too bright at 0
                         (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                         (1.0, 0.0, 0.0)),  # no green at 1

                 'blue': ((0.0, 0.0, 0.0),  # no blue at 0
                          (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                          (1.0, 0.0, 0.0))  # no blue at 1
                 }

        # Create the colormap using the dictionary
        greenRedCMap = colors.LinearSegmentedColormap('GreenRedCMap', cdict)

        plt.imshow(v_values, cmap=greenRedCMap, interpolation='nearest')
        plt.xticks(np.arange(len(v_values)), labels=np.arange(len(v_values)))
        plt.yticks(np.arange(len(v_values)), labels=np.arange(len(v_values)))

        # Loop over data dimensions and create text annotations.
        for i in range(len(v_values)):
            for j in range(len(v_values)):
                if (i,j) == self.treasure or (i,j) == self.snakepit:
                    text = "ABS"
                    weight = 'bold'
                elif (i,j) in self.obstacles:
                    text = "WALL"
                    weight = 'bold'
                elif plot_policy:
                    text = f"{v_values[i, j]:.0f}\n{self.actions[self.policy[i,j]]}"
                    weight = 'normal'
                else:
                    text = f"{v_values[i, j]:.0f}"
                    weight = 'normal'
                plt.text(j, i, text, ha="center", va="center", color="black", weight=weight)

        plt.title(title)
        plt.colorbar()
        plt.tight_layout()

        ax = plt.gca()
        # Minor ticks
        ax.set_xticks(np.arange(-.5, 9, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 9, 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=3)

        plt.show()


