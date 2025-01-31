import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

# Define a first-order continuous-time system in state-space representation
class FirstOrderSystemEnv(gym.Env):
    def __init__(self):
        super(FirstOrderSystemEnv, self).__init__()
        
        # Define state-space representation dx/dt = Ax + Bu
        self.A = np.array([[-1]])  # Decay factor
        self.B = np.array([[1]])   # Input gain
        self.dt = 0.1               # Discretization step
        
        # State and action spaces
        self.observation_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        self.state = np.array([0], dtype=np.float32)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Discretized update using Euler integration
        self.state = self.state + (self.A @ self.state + self.B @ action) * self.dt
        self.state = self.state.astype(np.float32)  # Ensure dtype consistency
        
        # Define a simple reward: try to drive state to zero
        reward = float(-np.abs(self.state[0]))  # Ensure reward is a float
        done = False
        truncated = False  # Ensure truncated is explicitly a boolean
        
        return self.state, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        self.state = np.random.uniform(-1, 1, size=(1,)).astype(np.float32)
        return self.state, {}  # Return state and an empty info dictionary

    def render(self):
        print(f"State: {self.state[0]:.3f}")

# Instantiate the environment
env = FirstOrderSystemEnv()
check_env(env)

env = DummyVecEnv([lambda: env])  # Wrap for compatibility

# Define the PPO model with Adaptive KL Penalty: hyperparameters 


ppo_model = PPO(
    policy=ActorCriticPolicy,
    env=env,
    learning_rate=3e-4, # Controls how much the model updates its policy per training step
    n_steps=2048, # The number of environment steps collected before updating the policy
    batch_size=64, # The number of samples used per training iteration
    n_epochs=15, # The number of times each batch of data is used to update the policy per PPO update ste
    gamma=0.995, # Discount factor: determines how much future rewards are valued compared to immediate rewards
    gae_lambda=0.95, # Generalized Advantage Estimation: balances bias vs. variance tradeoffin advantage estimation
    clip_range=0.2, # The maximum ratio between the new and old policy probabilities
    ent_coef=0.01, # Encourages exploration by adding entropy to the loss function
    target_kl=0.01,  # Target KL divergence: ensures updates do not change the policy too aggressively
    verbose=1
)

# Train the PPO agent
ppo_model.learn(total_timesteps=100000) # Defines how many steps the PPO agent trains


# Test the trained model and collect data
obs = env.reset()
states_rl = []
time_steps = []

for t in range(100):
    states_rl.append(obs[0][0])  # Store the RL-controlled state
    time_steps.append(t)
    
    action = ppo_model.predict(obs)
    obs, reward, done, truncated = env.step(action)
    obs = obs.astype(np.float32)  # Ensure dtype consistency
    
    if done or truncated:
        obs = env.reset()

# Generate a typical step response for a first-order system
def first_order_response(t, tau=1):
    return 1 - np.exp(-np.array(t) / tau)

time_steps = np.array(time_steps)
states_step_response = first_order_response(time_steps * env.get_attr('dt')[0])

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(time_steps, states_step_response, label="Typical Step Response", linestyle='dashed')
plt.plot(time_steps, states_rl, label="RL-Based Response")
plt.xlabel("Time Step")
plt.ylabel("State Value")
plt.title("Comparison of First-Order System Step Response vs. RL-Controlled Response")
plt.legend()
plt.grid()
plt.show()
   