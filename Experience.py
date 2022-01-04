class Experience(object):
    """
    Experience is a class that stores the state, action, reward, and next state
    """
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

    def __str__(self):
        return "[{}, {}, {}, {}]".format(self.state, self.action, self.reward, self.next_state)