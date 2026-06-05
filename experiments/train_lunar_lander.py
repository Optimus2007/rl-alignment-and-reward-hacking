import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import sys

# Connect to our custom wrappers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.wrappers import HoverExploitWrapper

def train_lander(env_type, total_timesteps=1500000):
    print(f"--- Starting LunarLander Experiment: {env_type} ---")
    
    # 1. Bulletproof absolute pathing for our new environment
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, f"../logs/lunar_lander/{env_type}/"))
    os.makedirs(log_dir, exist_ok=True)
    
    # 2. Initialize the LunarLander environment
    base_env = gym.make("LunarLander-v3")
    env = Monitor(base_env, log_dir)
    
    # 3. Apply the trap if we are testing the exploit
    if env_type == "hover_exploit":
        env = HoverExploitWrapper(env)
    # If env_type is "default", we let it use the normal rules
    
    # 4. The "True Objective" Evaluation Environment
    # We NEVER wrap this. We want to see how the agent performs 
    # under the real rules (crashing is bad, fuel costs points).
    eval_env = Monitor(gym.make("LunarLander-v3"))
    
    # 5. Setup the Evaluation Callback (Spaced out for 1.5M steps)
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=log_dir,
        log_path=log_dir, 
        eval_freq=50000,       # Increased from 10,000
        n_eval_episodes=5,
        deterministic=True, 
        render=False
    )
    
    # 6. Initialize standard PPO. 
    # We drop the high ent_coef here because dense rewards (like our hover +1) 
    # provide enough gradient that violent random exploration isn't necessary.
    model = PPO("MlpPolicy", env, verbose=0)
    
    # 7. Execute the 1.5 Million Step Training Loop
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    print(f"--- Finished LunarLander Experiment: {env_type} ---\n")
    
    env.close()

if __name__ == "__main__":
    # We only need two parallel universes for this experiment
    train_lander("default")
    train_lander("hover_exploit")