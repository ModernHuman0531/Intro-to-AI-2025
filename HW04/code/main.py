from BanditEnv import BanditEnv
from Agent import Agent
import matplotlib.pyplot as plt

def part3():
    """
    Run the following experiments 2000 times independently
    1. Construct a 10-armed bandit environment from part1
    2. Construct an agent with epsilon-greedy eploration from part2
    3. Run the agent in the environment for 1000 steps
    4. Compare a greedy method(𝜖 = 0) with two 𝜖-greedy method(𝜖 = 0.1 and 𝜖 = 0.01), 
    and plot two curves:
        a. The average reward of the agent over time
        b. The percentage of optimal action selection over time
    """
    # Set parameters
    times = 2000
    steps = 1000
    epsilons = [0, 0.01, 0.1]

    for i in range(len(epsilons)):
        # # Create lists to store the average reward and the optinal action selection
        avg_reward, optimal_action = [0]*1000, [0]*1000
        # Run the agent in the environment for 1000 steps
        # Set the begin to 1 to correspond the step 
        for time in range(times):
            # Create the environment
            Env = BanditEnv(10, stationary=True)
            # Create the agent
            agent_i = Agent(10, epsilon=epsilons[i])
            # Run the agent in the environment for 1000 steps
            for step in range(steps):
                # Select the action
                action = agent_i.select_action()
                # Take the action in the state to check the reward
                reward = Env.step(action)
                # Update the q_values in that action
                agent_i.update_q(action, reward)
                # Update the optimal action count 
                optimal_action[step] += 1 if action == Env.means.index(max(Env.means)) else 0
                # Update the reward of the step
                avg_reward[step] += reward
            # Reset the environment
            Env.reset()
            # Reset the agent
            agent_i.reset()
        # Calculate the average reward and the optimal action selection percentage
        optimal_action = [optimal_action[i]*100/times for i in range(steps)]
        avg_reward = [avg_reward[i]/times for i in range(steps)]
                
        # Draw the line of the average reward and the optimal action selection
        total_step = range(1, steps+1)
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(total_step, avg_reward, label="Average Reward")
        plt.xlabel("Steps")
        plt.ylabel("Average Rewards")
        plt.title(f"Average Reward with epsilon = {epsilons[i]}")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(total_step, optimal_action, label="Optimal Action Selection Percentage")
        plt.xlabel("Steps")
        plt.ylabel("Optimal Action Selection Percentage")
        plt.title(f"Optimal Action Selection Percentage with epsilons = {epsilons[i]}")
        plt.legend()
        plt.tight_layout()
        # Save the figure
        plt.savefig(f"part3_epsilons_{epsilons[i]}.png")
        plt.show()


def part5():
    """
    Need to perform the following experiments 2000 times independently
    1. Construct a 10-armed bandit environment with non-stationary reward distribution
    2. Construct an agent with epsilon-greedy eploration
    3. Run the agent in the environment for 10000 steps 
    """


        
if __name__ == "__main__":
    part3()
    #part5()
    #part7()
