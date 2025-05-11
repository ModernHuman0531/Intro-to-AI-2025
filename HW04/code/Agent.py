import random
class Agent:
    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon
        # Build a Q table to store the estimation
        self.q_table = [0] * self.k
        # Build a list to store the number of times each action has been taken
        self.action_count = [0] * self.k
    def select_action(self):
        """
        Choose an action based on the estimated expected reward for each action
        1. Apply epsilon greedy strategy
        2. Choose the action with the highest estimated expected reward
        3. If the random number is less than epsilon, choose a random action
        4. If the random number is greater than epsilon, choose the action with the highest estimated expected reward
        5. Return the action
        """
        # Generate a random number between 0 and 1
        random_number = random.random()
        # Design the action in the range of 0 to k-1
        if random_number < self.epsilon:
            action = random.randint(0, self.k - 1)
        else:
            # max(self.q_table) will return the max value in the q_table, .index will return the index of the first max value
            action = self.q_table.index(max(self.q_table))
        return action
    def update_q(self, action, reward):
        """
        Update the estimated expexted reward of the chossen action
        1. The estimated expected reward is updated using the formula:
        Q(a) = Q(a) + (reward - Q(a)) / N
        2. N is the number of times the action has been taken
        3. N is updated by adding 1 to the action count
        """
        # Update the action count by 1
        self.action_count[action] += 1
        # Update the estimated expexted reward using the formula
        # Q(a) = Q(a) + (reward - Q(a)) / N
        self.q_table[action] += (reward - self.q_table[action])/self.action_count[action] 

    def reset(self):
        """
        Reset the agent
        """
        self.q_table = [0] * self.k
        self.action_count = [0] * self.k