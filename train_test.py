# train_rl_agent.py
import sys
import os
import numpy as np
import time
import argparse
from collections import deque
import pickle

# Set Pyglet configurations for macOS
os.environ['PYGLET_SHADOW_WINDOW'] = '0'

# Add the parent directory to sys.path
sys.path.append('.')

# Import dependencies
import gymnasium as gym
from envs.atc.atc_gym import AtcGym
import envs.atc.model as model
from envs.atc.scenarios import SimpleScenario
import envs  # Required to register the environment

class SimplePPOAgent:
    """
    A very simplified version of PPO (Proximal Policy Optimization) for demonstration.
    This is not a full implementation - just a dummy agent for testing.
    """
    def __init__(self, state_dim, action_dim, learning_rate=0.0003):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Simple linear policy (for demonstration)
        self.policy_weights = np.random.randn(state_dim, action_dim) * 0.1
        self.value_weights = np.random.randn(state_dim, 1) * 0.1
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
    def select_action(self, state):
        """Simple stochastic policy implementation"""
        # Linear layer for mean
        action_mean = np.tanh(np.dot(state, self.policy_weights))
        
        # Add exploration noise
        action = action_mean + np.random.normal(0, 0.2, size=self.action_dim)
        action = np.clip(action, -1.0, 1.0)
        
        # Calculate value (not used in this simplified version)
        value = np.dot(state, self.value_weights)[0]
        
        # Store experience
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        
        return action
    
    def update_policy(self, final_value=0):
        """Very simplified policy update for demonstration"""
        # Process rewards with simple returns calculation
        returns = self._compute_returns(final_value, gamma=0.99)
        
        # Simplified update (just move weights in direction of positive rewards)
        for i in range(len(self.states)):
            state = self.states[i]
            action = self.actions[i]
            ret = returns[i]
            
            # Update policy (simplified)
            update = self.learning_rate * ret * np.outer(state, action)
            self.policy_weights += update
            
            # Update value function (simplified)
            self.value_weights += self.learning_rate * ret * state.reshape(-1, 1)
        
        # Clear experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
    
    def _compute_returns(self, final_value, gamma):
        """Compute returns with discount factor gamma"""
        returns = []
        R = final_value
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns

    def add_reward(self, reward):
        """Store a reward for the most recent action"""
        self.rewards.append(reward)
    
    def save(self, filename):
        """Save the agent's policy weights"""
        data = {
            'policy_weights': self.policy_weights,
            'value_weights': self.value_weights
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to {filename}")
    
    def load(self, filename):
        """Load the agent's policy weights"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.policy_weights = data['policy_weights']
        self.value_weights = data['value_weights']
        print(f"Agent loaded from {filename}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a simple RL agent on the ATC environment')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--max-steps', type=int, default=600, help='Maximum steps per episode')
    parser.add_argument('--eval-interval', type=int, default=10, help='Evaluate every N episodes')
    parser.add_argument('--save-path', type=str, default='agent.pkl', help='Path to save the trained agent')
    parser.add_argument('--load-path', type=str, default=None, help='Path to load a trained agent')
    parser.add_argument('--render-eval', action='store_true', help='Render during evaluation episodes')
    return parser.parse_args()

def evaluate_agent(agent, env, num_episodes=3, max_steps=600, render=False):
    """Evaluate the agent's performance"""
    total_rewards = []
    
    # Set render mode based on argument
    old_render_mode = env.render_mode
    if render:
        env.render_mode = 'human'
    else:
        env.render_mode = 'headless'
    
    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        while not done and step < max_steps:
            action = agent.select_action(state)
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
            if render:
                env.render()
                time.sleep(0.05)
            
            step += 1
        
        total_rewards.append(episode_reward)
    
    # Restore original render mode
    env.render_mode = old_render_mode
    
    return np.mean(total_rewards)

def main():
    args = parse_args()
    
    # Create environment in headless mode for training
    sim_params = model.SimParameters(1.0, discrete_action_space=False)
    env = AtcGym(airplane_count=1, sim_parameters=sim_params, 
                 scenario=SimpleScenario(), render_mode='headless')

    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create the agent
    agent = SimplePPOAgent(state_dim, action_dim)
    
    # Load pretrained agent if specified
    if args.load_path is not None:
        agent.load(args.load_path)
    
    # Track rewards for statistics
    rewards_history = deque(maxlen=100)
    best_eval_reward = -float('inf')
    
    # Training loop
    start_time = time.time()
    for episode in range(args.episodes):
        # Reset environment
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        # Run episode
        for step in range(args.max_steps):
            # Select action based on current state
            action = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, _, _ = env.step(action)
            
            # Store reward
            agent.add_reward(reward)
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update policy at end of episode
        agent.update_policy()
        
        # Store reward for statistics
        rewards_history.append(total_reward)
        
        # Print progress
        if episode % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode}/{args.episodes} | "
                  f"Reward: {total_reward:.2f} | "
                  f"Average (100): {np.mean(rewards_history):.2f} | "
                  f"Time: {elapsed:.2f}s")
        
        # Evaluate agent periodically
        if episode % args.eval_interval == 0:
            eval_reward = evaluate_agent(agent, env, render=args.render_eval)
            print(f"Evaluation after episode {episode}: {eval_reward:.2f}")
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(args.save_path)
                print(f"New best model saved with reward {eval_reward:.2f}")
    
    # Final evaluation
    final_reward = evaluate_agent(agent, env, render=args.render_eval)
    print(f"Final evaluation: {final_reward:.2f}")
    
    # Save final model
    agent.save(args.save_path + '.final')
    print(f"Training completed. Final model saved.")
    
    env.close()

if __name__ == "__main__":
    main()