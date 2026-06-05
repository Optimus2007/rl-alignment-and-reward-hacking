import numpy as np
import matplotlib.pyplot as plt
import os

def load_eval_data(env_type):
    """Loads the evaluation data saved by SB3's EvalCallback."""
    # Go up one directory from 'analysis' to find 'logs'
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../logs/mountain_car/{env_type}/evaluations.npz"))
    
    if not os.path.exists(file_path):
        print(f"Could not find data for {env_type} at {file_path}")
        return None, None, None, None
        
    data = np.load(file_path)
    timesteps = data['timesteps']
    
    # Calculate means across the 5 evaluation episodes we ran at each checkpoint
    mean_rewards = np.mean(data['results'], axis=1)
    mean_lengths = np.mean(data['ep_lengths'], axis=1)
    
    # In MountainCar, reaching the flag stops the episode before the 200-step limit.
    # Therefore, any episode length < 200 is a success.
    success_rates = np.mean(data['ep_lengths'] < 200, axis=1) * 100
    
    return timesteps, mean_rewards, mean_lengths, success_rates

def generate_plots():
    print("Loading data and generating plots...")
    
    # 1. Load the data for all three parallel universes
    agents = {
        "Default (-1 per step)": {"type": "default", "color": "blue"},
        "Pure Sparse (+1 at goal)": {"type": "sparse", "color": "red"},
        "Heavily Shaped (Energy)": {"type": "shaped", "color": "green"}
    }
    
    # Set up a beautiful 1x3 grid for our research report
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("MountainCar: The Density-Exploration Trade-off", fontsize=16, fontweight='bold')
    
    for label, config in agents.items():
        t, rewards, lengths, success = load_eval_data(config["type"])
        
        if t is not None:
            # Plot 1: True Task Return
            axs[0].plot(t, rewards, label=label, color=config["color"], linewidth=2)
            # Plot 2: Episode Length
            axs[1].plot(t, lengths, label=label, color=config["color"], linewidth=2)
            # Plot 3: Success Rate
            axs[2].plot(t, success, label=label, color=config["color"], linewidth=2)

    # Format Plot 1
    axs[0].set_title("True Task Return")
    axs[0].set_xlabel("Training Timesteps")
    axs[0].set_ylabel("Average Reward (Unmodified Env)")
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend()

    # Format Plot 2
    axs[1].set_title("Episode Length (Efficiency)")
    axs[1].set_xlabel("Training Timesteps")
    axs[1].set_ylabel("Steps to Complete (Max 200)")
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend()

    # Format Plot 3
    axs[2].set_title("Success Rate")
    axs[2].set_xlabel("Training Timesteps")
    axs[2].set_ylabel("Success Percentage (%)")
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].legend()

    plt.tight_layout()
    
    # Save the figure so you can put it in your technical report later
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs/mountain_car/learning_curves.png"))
    plt.savefig(save_path)
    print(f"Plot saved successfully to: {save_path}")
    
    # Display the plot on your screen
    plt.show()

if __name__ == "__main__":
    generate_plots()