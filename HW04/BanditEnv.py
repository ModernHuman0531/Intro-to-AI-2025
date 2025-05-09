import random
class BanditEnv:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.means = None
        self.history = {"actions": [], "rewards": []}


    def reset(self):
        """
        Reset the game envitonment and the history
        """
        self.history = {"actions": [], "rewards": []}
        self.means = None



    def step(self, action):
        """
        Take an action and return the reward
        1. The reward distribution for each arm should be a Gaussian distribution with 
        variance 1 and randomally generated mean
        2. The mean should be generated from a standard normal distribution 
        """
        # Generate the means for each arm in standard normal distribution
        if self.means == None:
            self.means = [random.gauss(0, 1) for i in range(self.n_arms)]
        # Get the reward for the action based on the mean
        reward = random.gauss(self.means[action], 1)
        # Updqate the history
        self.history["actions"].append(action)
        self.history["rewards"].append(reward)
        return reward
        
    def export_history(self):
        """
        Export the action history and the reward history
        """
        return self.history
