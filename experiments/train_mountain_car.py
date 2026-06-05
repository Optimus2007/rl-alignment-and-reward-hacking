import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import sys

# Add the repo root to Python's path so we can import our `src` package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.wrappers import PureSparseRewardWrapper, HeavilyShapedEnergyWrapper

def train_agent(env_type, total_timesteps=200000):
    print(f"--- Starting Experiment: {env_type} ---")
    
    # 1. Bulletproof absolute pathing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, f"../logs/mountain_car/{env_type}/"))
    os.makedirs(log_dir, exist_ok=True)
    
    # 2. Initialize the standard environment
    # We use Monitor to keep track of true episode lengths and returns
    base_env = gym.make("MountainCar-v0")
    env = Monitor(base_env, log_dir)
    
    # 3. Apply the appropriate Reward Wrapper based on the experiment
    if env_type == "sparse":
        env = PureSparseRewardWrapper(env)
    elif env_type == "shaped":
        env = HeavilyShapedEnergyWrapper(env)
    # If env_type is "default", we don't wrap it!
    
    # 4. Create the Evaluation Environment (The "True Objective" test)
    # This must be pure and unmodified, so we do NOT wrap it.
    eval_env = Monitor(gym.make("MountainCar-v0"))
    
    # 5. Setup the Evaluation Callback
    # It will pause every 10,000 steps and test the agent 5 times
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=log_dir,
        log_path=log_dir, 
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True, 
        render=False
    )
    
   # 6. Initialize the PPO Brain (with aggressive hyperparameters)
    model = PPO("MlpPolicy", env, 
                verbose=0, 
                ent_coef=0.01,
                learning_rate=0.001) # Increased from default 0.0003
    
    # 7. Execute the Training Loop (Give it more time to learn!)
    model.learn(total_timesteps=500000, callback=eval_callback)
    print(f"--- Finished Experiment: {env_type} ---\n")
    
    env.close()

if __name__ == "__main__":
    # Run the three parallel universes
    train_agent("default")
    train_agent("sparse")
    train_agent("shaped")