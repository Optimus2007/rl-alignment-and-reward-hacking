import gymnasium as gym
import numpy as np

class PureSparseRewardWrapper(gym.RewardWrapper):
    """
    Hypothesis 1a: The agent receives 0 reward at all times, 
    unless it reaches the flag (position >= 0.5), where it receives +1.
    """
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        # We ignore the default 'reward' (-1) passed by the base environment
        # We must extract the position (state index 0) to check for success
        position = self.unwrapped.state[0] 
        
        if position >= 0.5:
            return 1.0  # The single needle in the haystack
        else:
            return 0.0  # The dark void

class HeavilyShapedEnergyWrapper(gym.RewardWrapper):
    """
    Hypothesis 1b: The agent is rewarded continuously for building 
    mechanical energy (kinetic + potential), encouraging the swing trick.
    """
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        # Extract both position and velocity from the physics engine
        position = self.unwrapped.state[0]
        velocity = self.unwrapped.state[1]
        
        # Calculate Potential Energy based on the hill's curve (sin wave)
        # Note: MountainCar uses a 3*x frequency for its hills
        potential_energy = np.sin(3 * position)
        
        # Calculate Kinetic Energy based on speed in either direction
        kinetic_energy = (velocity ** 2)
        
        # We scale the reward so it isn't massive, which keeps PPO stable
        total_energy_reward = (potential_energy + kinetic_energy) * 10 
        
        return total_energy_reward

class HoverExploitWrapper(gym.Wrapper):
    """
    Hypothesis 2: The Alignment Trap. 
    We intercept the step function to read the observation array directly,
    penalizing touching the ground (-100) and continuously rewarding hovering (+1).
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # 1. Let the Box2D physics engine take the step first
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # 2. Extract our variables directly from the fresh observation array
        y_pos = observation[1]
        left_leg_contact = observation[6]
        right_leg_contact = observation[7]
        
        # 3. The Floor is Lava Trap
        if left_leg_contact == 1.0 or right_leg_contact == 1.0:
            modified_reward = -100.0  # Massive penalty for landing
            
        # 4. The Hovering Exploit
        elif y_pos > 0:
            modified_reward = 1.0     # Infinite point farming in the air
            
        # Fallback
        else:
            modified_reward = 0.0
            
        # 5. Return the exact same physics, but with our hacked reward
        return observation, modified_reward, terminated, truncated, info