from BanditEnv import BanditEnv
from Agent import Agent
import matplotlib.pyplot as plt

def part3():
    """
    Run the part3 of the project
    1. Construct a 10-armed bandit environment from part1
    2. Construct an agent with epsilon-greedy eploration from part2
    3. Run the agent in the environment for 1000 steps
    4. Compare a greedy method(𝜖 = 0) with two 𝜖-greedy method(𝜖 = 0.1 and 𝜖 = 0.01), 
    and plot two curves:
        a. The average reward of the agent over time
        b. The percentage of optimal action selection over time
    """
    # Set parameters
    times = 1000
    epsilons = [0, 0.01, 0.1]
    Env = BanditEnv(10)

    for i in range(len(epsilons)):
        # Create the agent
        agent_i = Agent(10, epsilons[i])
        # # Create lists to store the average reward and the optinal action selection
        avg_reward, optimal_action = [], []
        # Create another variables named optimal_action_count and total_reward to keep track of the number
        optimal_action_count, total_reward = 0, 0
        # Run the agent in the environment for 1000 steps
        # Set the begin to 1 to correspond the step 
        for step in range(1, times+1):
            # Select the action
            action = agent_i.select_action()
            # Take the action in the state to check the reward
            reward = Env.step(action)
            # Update the q_values in that action
            agent_i.update_q(action, reward)
            # Update the optimal action count
            optimal_action_count += 1 if action == Env.means.index(max(Env.means)) else 0
            # Update the total reward
            total_reward += reward
            # Append the average reward and and tge optional action percentage into the list
            optimal_action.append(optimal_action_count*100/step)
            avg_reward.append(total_reward/step)
        # Draw the line of the average reward and the optimal action selection
        steps = range(1, times+1)
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(steps, avg_reward, label="Average Reward")
        plt.xlabel("Steps")
        plt.ylabel("Average Rewards")
        plt.title(f"Average Reward with epsilon = {epsilons[i]}")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(steps, optimal_action, label="Optimal Action Selection Percentage")
        plt.xlabel("Steps")
        plt.ylabel("Optimal Action Selection Percentage")
        plt.title(f"Optimal Action Selection Percentage with epsilons = {epsilons[i]}")
        plt.legend()
        plt.tight_layout()
        # Save the figure
        plt.savefig(f"part3_epsilons_{epsilons[i]}.png")
        plt.show()

        # Clean up the environment and the agent
        Env.reset()
        agent_i.reset()

            


        
if __name__ == "__main__":
    part3()
    #part5()
    #part7()
