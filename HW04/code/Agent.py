import random
class Agent:
    """
    1. If the alpha is None, the agent will be same as part2(), and the update_q method will use 
    sample average method
    2. If the alpha is not None, the agent's update_q method will use the step-size update method
    3. The step-size update method will use the formula:
    Q(a) = Q(a) + alpha*(reward - Q(a)), alpha is the step-size
    """
    def __init__(self, k, epsilon, alpha=None):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_table = [0] * self.k
        self.action_count = [0] * self.k
    def select_action(self):
        # random.random will pick a number between 0 to 1
        random_number = random.random()

        # If the randomnumber is less than epsilon, do the exloration
        if random_number < self.epsilon:
            action = random.radint(0, self.k-1)
            # Else, do the exploitation, which is choose the action with the highest q value
        else:
            action = self.q_table.index(max(self.q_table))
        return action
    def update_q(self, action, reward):
        # If alpha is None, use the sample average method
        if self.alpha == None:
            # Update the action count by 1
            self.action_count[action] += 1
            # Update the reward with the sample average method
            # The formula is Q(a) = Q(a) + (reward - Q(a)) / N
            self.q_table[action] += (reward - self.q_table[action]) / self.action_count[action]
        else:
            # Update the reward with the step-size update method
            # The formula is Q(a) = Q(a) + alpha*(reward-Q(a))
            self.q_table[action] += self.alpha*(reward-self.q_table[action])

    def reset(self):
        self.q_table = [0] * self.k
        self.action_count = [0] * self.k   

