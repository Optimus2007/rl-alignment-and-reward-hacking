import numpy as np
import matplotlib.pyplot as plt
import os

def load_eval_data(env_type):
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../logs/lunar_lander/{env_type}/evaluations.npz"))
    
    if not os.path.exists(file_path):
        print(f"Could not find data for {env_type} at {file_path}")
        return None, None, None, None
        
    data = np.load(file_path)
    timesteps = data['timesteps']
    
    mean_rewards = np.mean(data['results'], axis=1)
    mean_lengths = np.mean(data['ep_lengths'], axis=1)
    
    
    success_rates = np.mean(data['results'] > 200, axis=1) * 100
    
    return timesteps, mean_rewards, mean_lengths, success_rates

def generate_plots():
    print("Loading data and generating LunarLander plots...")
    
    agents = {
        "Default (Intended Behavior)": {"type": "default", "color": "blue"},
        "Hover Exploit (Reward Hacked)": {"type": "hover_exploit", "color": "red"}
    }
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("LunarLander: Intentional Reward Hacking (The Alignment Problem)", fontsize=16, fontweight='bold')
    
    for label, config in agents.items():
        t, rewards, lengths, success = load_eval_data(config["type"])
        
        if t is not None:
            axs[0].plot(t, rewards, label=label, color=config["color"], linewidth=2)
            axs[1].plot(t, lengths, label=label, color=config["color"], linewidth=2)
            axs[2].plot(t, success, label=label, color=config["color"], linewidth=2)

    axs[0].set_title("True Task Return (Unmodified Rules)")
    axs[0].set_xlabel("Training Timesteps")
    axs[0].set_ylabel("Average True Reward")
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend()

    axs[1].set_title("Episode Length (Behavior)")
    axs[1].set_xlabel("Training Timesteps")
    axs[1].set_ylabel("Steps Before Termination")
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend()

    axs[2].set_title("Success Rate (Score > 200)")
    axs[2].set_xlabel("Training Timesteps")
    axs[2].set_ylabel("Success Percentage (%)")
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].legend()

    plt.tight_layout()
    
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs/lunar_lander/reward_hacking_curves.png"))
    plt.savefig(save_path)
    print(f"Plot saved successfully to: {save_path}")
    plt.show()

if __name__ == "__main__":
    generate_plots()