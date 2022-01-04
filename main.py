from GridWorld import GridWorld
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(10)


def plot_stats(x, y, x_label, y_label,  labels, title="", std_y=None, y_scale="linear", figsize=(10,5)):
    """
    Plot the given data.
    :param x: Array of x values.
    :param y:  List of array of y values.
    :param x_label:  Label of the x axis.
    :param y_label: Label of the y axis.
    :param title: Title of the plot.
    :param labels: List of labels
    :param std_y: List of standard deviation of the y values.
    :param y_scale: Scale of the y axis.
    """
    plt.figure(figsize=figsize)
    for i in range(len(y)):
        plt.plot(x, y[i], linestyle='solid', lw=2, label=labels[i])
        if std_y:
            plt.fill_between(x, y[i]-std_y[i], y[i]+std_y[i], alpha=0.3)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(linestyle='--', linewidth=1)
    plt.yscale(y_scale)
    plt.show()


def main():
    # initialize the gridworld environment
    obstacles = [(1,2), (1,3), (1,4), (1,5), (1,6), (2,6), (3,6), (4,6), (5,6), (7,1), (7,2), (7,3), (7,4)]
    treasure = (8,8)
    snakepit = (6,5)
    grid = GridWorld(9, obstacles, treasure, snakepit)

    # plot gridworld
    grid.plot_gridworld(
        title="Rewards of the Gridworld"
    )

    # -------------------------------------------------------------------------------#
    # ----------------------------- SARSA method ------------------------------------#
    # -------------------------------------------------------------------------------#
    num_episodes = 10000
    q_values_sarsa, stats_sarsa = grid.sarsa_algorithm(num_episodes=num_episodes, epsilon=0.2, alpha=.5, return_stats=True)
    v_values_sarsa = grid.compute_v_values(q_values_sarsa)
    grid.plot_value_function(v_values_sarsa, plot_policy=True
                             ,title="Value function and optimal policy of the Gridworld using SARSA method"
                             )

    # -------------------------------------------------------------------------------#
    # ----------------------------- Q-learning method -------------------------------#
    # -------------------------------------------------------------------------------#
    q_values_ql, stats_ql = grid.q_learning_algorithm(num_episodes=num_episodes, epsilon=0.2, alpha=.5, return_stats=True)
    v_values_ql = grid.compute_v_values(q_values_ql)
    grid.plot_value_function(v_values_ql, plot_policy=True
                             , title="Value function and optimal policy of the Gridworld using Q-learning method"
                             )

    # -------------------------------------------------------------------------------#
    # --------------------------- Monte Carlo method --------------------------------#
    # -------------------------------------------------------------------------------#
    # compute the value function using the Monte Carlo method
    v_values_mc = grid.monte_carlo_equiprobable_policy_evaluation(num_episodes=1000, every_visit=False)
    grid.plot_value_function(v_values_mc
                             , title="Value function of the Gridworld using first visit Monte Carlo method"
                             )

    v_values_mc = grid.monte_carlo_equiprobable_policy_evaluation(num_episodes=1000, every_visit=True)
    grid.plot_value_function(v_values_mc
                             , title="Value function of the Gridworld using every visit Monte Carlo method"
                             )

    # plot statistics
    plot_episodes = range(0, num_episodes, 150)
    plot_stats(
        x=plot_episodes,
        y=[stats_sarsa['episode_steps'][plot_episodes], stats_ql['episode_steps'][plot_episodes]],
        x_label="Episode",
        y_label="Steps",
        title="Number of steps per episode",
        labels=["Sarsa", "Q-learning"],
        y_scale="log"
    )
    plot_stats(
        x=plot_episodes,
        y=[stats_sarsa['avg_td'][plot_episodes], stats_ql['avg_td'][plot_episodes]],
        x_label="Episode",
        y_label="Avg TD error",
        title="Average Temporal Differencing error per episode",
        labels=["Sarsa", "Q-learning"],
        std_y=[stats_sarsa['std_td'][plot_episodes], stats_ql['std_td'][plot_episodes]]
    )


if __name__ == '__main__':
    main()
