import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.wrappers import HoverExploitWrapper

def train_lander(env_type, total_timesteps=1500000):
    print(f"--- Starting LunarLander Experiment: {env_type} ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, f"../logs/lunar_lander/{env_type}/"))
    os.makedirs(log_dir, exist_ok=True)
    
    base_env = gym.make("LunarLander-v3")
    env = Monitor(base_env, log_dir)
    
    if env_type == "hover_exploit":
        env = HoverExploitWrapper(env)
    # If env_type is "default", we let it use the normal rules
    
    eval_env = Monitor(gym.make("LunarLander-v3"))
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=log_dir,
        log_path=log_dir, 
        eval_freq=50000,       
        n_eval_episodes=5,
        deterministic=True, 
        render=False
    )
    
    model = PPO("MlpPolicy", env, verbose=0)
    
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    print(f"--- Finished LunarLander Experiment: {env_type} ---\n")
    
    env.close()

if __name__ == "__main__":
    train_lander("default")
    train_lander("hover_exploit")