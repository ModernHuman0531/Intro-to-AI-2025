import random
class BanditEnv:
    """
    Create a non-stationary bandit environment that the mean reward 
    of each action changes over time, in this case, the mean reward will change in each step
    1. Modify the BanditEnv class to include a non-stationary reward distribution
    2. The true mean of each action will change in every steps
    3. The true mean reward of each action will take independent random
    walks by adding a normally distributed increment with mean 0 and 
    standard deviation 0.01 to all the true mean rewards
    """
    def __init__(self, n_arms, stationary = True):
        self.n_arms = n_arms
        self.stationary = stationary
        self.means = None
        self.history = {"actions": [], "rewards": []}

    def reset(self):
        """
        Reset the game envitonment and the history
        """
        self.history = {"actions": [], "rewards": []}
        self.means = None

    def step(self, action):
        # If is stationary, the mean reward will not change
        if self.means == None:
            self.means = [random.gauss(0, 1) for i in range(self.n_arms)]
        if self.stationary == True:
            # Get the reward for the action based on the mean
            reward = random.gauss(self.means[action], 1)
        else:
            self.means = [self.means[i] + random.gauss(0, 0.01) for i in range(self.n_arms)]
            reward = random.gauss(self.means[action], 1)

            # Update the history
        self.history["actions"].append(action)
        self.history["rewards"].append(reward)
        return reward
    def export_history(self):
        """
        Export the action history and the reward history
        """
        return self.history
        
