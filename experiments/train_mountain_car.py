import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.wrappers import PureSparseRewardWrapper, HeavilyShapedEnergyWrapper

def train_agent(env_type, total_timesteps=200000):
    print(f"--- Starting Experiment: {env_type} ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, f"../logs/mountain_car/{env_type}/"))
    os.makedirs(log_dir, exist_ok=True)
    
    base_env = gym.make("MountainCar-v0")
    env = Monitor(base_env, log_dir)
    
    if env_type == "sparse":
        env = PureSparseRewardWrapper(env)
    elif env_type == "shaped":
        env = HeavilyShapedEnergyWrapper(env)
    
    eval_env = Monitor(gym.make("MountainCar-v0"))
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=log_dir,
        log_path=log_dir, 
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True, 
        render=False
    )
    
    model = PPO("MlpPolicy", env, 
                verbose=0, 
                ent_coef=0.01,
                learning_rate=0.001) 
    
    model.learn(total_timesteps=500000, callback=eval_callback)
    print(f"--- Finished Experiment: {env_type} ---\n")
    
    env.close()

if __name__ == "__main__":
    train_agent("default")
    train_agent("sparse")
    train_agent("shaped")