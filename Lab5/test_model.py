"""
Author: Chi-An Chen
Date: 2025-08-02
Description: Evaluating DQN model 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import os
from collections import deque
import argparse
import time
import ale_py
gym.register_envs(ale_py)

class DQN(nn.Module):
    """
    Enhanced Deep Q Network with improved architecture
    - Added dropout for regularization
    - Increased network capacity
    - Added batch normalization for stable training
    """
    def __init__(self, input_dim=4, num_actions=2, hidden_dim=256, dropout_rate=0.1):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim//2, num_actions)
        )

    def forward(self, x):
        return self.network(x)

class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture that separates value and advantage streams
    This helps with learning by explicitly separating the estimation of 
    state values and action advantages
    """
    def __init__(self, input_dim=4, num_actions=2, hidden_dim=256, dropout_rate=0.1):
        super(DuelingDQN, self).__init__()
        
        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_actions)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantage using dueling formula
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class AtariCNNDQN(nn.Module):
    """CNN-based DQN for Atari environments"""
    def __init__(self, num_actions, input_channels=4, dropout_rate=0.1):
        super(AtariCNNDQN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        
        # Calculate feature size dynamically
        self.feature_size = self._get_conv_output_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        self.dropout = nn.Dropout(dropout_rate)

    def _get_conv_output_size(self):
        """Calculate the output size of convolutional layers"""
        x = torch.zeros(1, 4, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.size()[1:]))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class AtariDuelingCNNDQN(nn.Module):
    """Dueling CNN DQN for Atari environments"""
    def __init__(self, num_actions, input_channels=4, dropout_rate=0.1):
        super(AtariDuelingCNNDQN, self).__init__()
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        
        # Calculate feature size dynamically
        self.feature_size = self._get_conv_output_size()
        # Shared feature layer
        self.feature_fc = nn.Linear(self.feature_size, 512)
        # Value stream
        self.value_fc = nn.Linear(512, 1)
        # Advantage stream
        self.advantage_fc = nn.Linear(512, num_actions)
        self.dropout = nn.Dropout(dropout_rate)

    def _get_conv_output_size(self):
        """Calculate the output size of convolutional layers"""
        x = torch.zeros(1, 4, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.size()[1:]))

    def forward(self, x):
        # Shared features
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        features = F.relu(self.feature_fc(x))
        features = self.dropout(features)
        
        # Value and advantage streams
        value = self.value_fc(features)
        advantage = self.advantage_fc(features)
        
        # Combine using dueling formula
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

class CartpolePreprocessor:
    """Enhanced preprocessor with state normalization for CartPole"""
    def __init__(self):
        # CartPole state bounds for normalization
        self.state_bounds = {
            'cart_pos': 2.4,
            'cart_vel': 3.0, 
            'pole_angle': 0.209,  # ~12 degrees
            'pole_vel': 3.0
        }
    
    def normalize_state(self, state):
        """Normalize state to improve learning stability"""
        normalized = np.array([
            state[0] / self.state_bounds['cart_pos'],
            state[1] / self.state_bounds['cart_vel'],
            state[2] / self.state_bounds['pole_angle'],
            state[3] / self.state_bounds['pole_vel']
        ])
        return np.clip(normalized, -1.0, 1.0)
    
    def reset(self, obs):
        return self.normalize_state(obs)
    
    def step(self, obs):
        return self.normalize_state(obs)

class AtariPreprocessor:
    """Preprocessor for Atari games"""
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        stacked = np.stack(self.frames, axis=0)
        return stacked

def load_model(model_path, device, env_name, model_type="standard"):
    """
    Load trained model with automatic architecture detection
    """
    print(f"Loading model from: {model_path}")
    
    # Detect environment type
    is_atari = "ALE/" in env_name or "Pong" in env_name or "Breakout" in env_name
    
    if is_atari:
        # Load Atari model
        env = gym.make(env_name)
        num_actions = env.action_space.n
        model = AtariDuelingCNNDQN(num_actions=num_actions).to(device)
        env.close()
        print(f"Loaded Atari DQN model with {num_actions} actions")
    else:
        # Load CartPole model
        env = gym.make(env_name)
        num_actions = env.action_space.n
        
        # Try to detect model type from checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'q_net_state_dict' in checkpoint:
            # Enhanced checkpoint format
            state_dict = checkpoint['q_net_state_dict']
            
            # Detect if it's dueling architecture
            has_value_stream = any('value_stream' in key for key in state_dict.keys())
            
            if has_value_stream:
                model = DuelingDQN(input_dim=4, num_actions=num_actions, hidden_dim=256).to(device)
                print("Loaded Dueling DQN model")
            else:
                model = DQN(input_dim=4, num_actions=num_actions, hidden_dim=256).to(device)
                print("Loaded Enhanced DQN model")
            
            model.load_state_dict(state_dict)
        else:
            # Legacy format - try both architectures
            try:
                model = DuelingDQN(input_dim=4, num_actions=num_actions, hidden_dim=256).to(device)
                model.load_state_dict(checkpoint)
                print("Loaded Dueling DQN model (legacy format)")
            except:
                try:
                    model = DQN(input_dim=4, num_actions=num_actions, hidden_dim=256).to(device)
                    model.load_state_dict(checkpoint)
                    print("Loaded Enhanced DQN model (legacy format)")
                except:
                    # Fallback to basic DQN
                    model = nn.Sequential(
                        nn.Linear(4, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, num_actions)
                    ).to(device)
                    model.load_state_dict(checkpoint)
                    print("Loaded Basic DQN model (legacy format)")
        
        env.close()
    
    model.eval()
    return model

def evaluate_cartpole(args):
    """Evaluate CartPole environment"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed) 
    torch.manual_seed(args.seed)

    # Create environment
    env = gym.make(args.env_name, render_mode="rgb_array")
    env.action_space.seed(args.seed)
    
    # Load model and preprocessor
    model = load_model(args.model_path, device, args.env_name)
    preprocessor = CartpolePreprocessor()

    os.makedirs(args.output_dir, exist_ok=True)

    total_rewards = []
    episode_lengths = []

    print(f"\nStarting evaluation for {args.episodes} episodes...")
    print("=" * 60)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0
        frames = []
        step_count = 0
        max_steps = args.max_steps if args.max_steps > 0 else float('inf')

        while not done and step_count < max_steps:
            # Render frame
            if args.save_video:
                frame = env.render()
                frames.append(frame)

            # Select action
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            # Take action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs)
            step_count += 1

        total_rewards.append(total_reward)
        episode_lengths.append(step_count)

        print(f"Episode {ep+1:3d}: Reward = {total_reward:6.1f}, Steps = {step_count:3d}")

        # Save video if requested
        if args.save_video and frames:
            out_path = os.path.join(args.output_dir, f"cartpole_ep{ep+1}.mp4")
            with imageio.get_writer(out_path, fps=30) as video:
                for frame in frames:
                    video.append_data(frame)
            print(f"                 Video saved: {out_path}")

    # Print summary statistics
    print("=" * 60)
    print("EVALUATION SUMMARY:")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.1f}")
    print(f"Min Reward: {np.min(total_rewards):.1f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Success Rate (>= 200 steps): {np.sum(np.array(episode_lengths) >= 200) / len(episode_lengths) * 100:.1f}%")
    print("=" * 60)

def evaluate_atari(args):
    """Evaluate Atari environment"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment
    env = gym.make(args.env_name, render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    # Load model and preprocessor
    model = load_model(args.model_path, device, args.env_name)
    preprocessor = AtariPreprocessor()

    os.makedirs(args.output_dir, exist_ok=True)

    total_rewards = []

    print(f"\nStarting evaluation for {args.episodes} episodes...")
    print("=" * 60)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0
        frames = []
        step_count = 0
        max_steps = args.max_steps if args.max_steps > 0 else 10000

        while not done and step_count < max_steps:
            # Render frame
            if args.save_video:
                frame = env.render()
                frames.append(frame)

            # Select action
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            # Take action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs)
            step_count += 1

        total_rewards.append(total_reward)
        print(f"Episode {ep+1:3d}: Reward = {total_reward:6.1f}, Steps = {step_count:4d}")

        # Save video if requested
        if args.save_video and frames:
            out_path = os.path.join(args.output_dir, f"atari_ep{ep+1}.mp4")
            with imageio.get_writer(out_path, fps=30) as video:
                for frame in frames:
                    video.append_data(frame)
            print(f"                 Video saved: {out_path}")

    # Print summary statistics
    print("=" * 60)
    print("EVALUATION SUMMARY:")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.1f}")
    print(f"Min Reward: {np.min(total_rewards):.1f}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Enhanced DQN Model Evaluation")
    
    # Required arguments
    parser.add_argument("--model-path", type=str, required=True, 
                       help="Path to trained model file (.pt)")
    
    # Environment arguments
    parser.add_argument("--env-name", type=str, default="CartPole-v1",
                       help="Environment name (e.g., CartPole-v1, ALE/Pong-v5)")
    
    # Evaluation arguments
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes to evaluate")
    parser.add_argument("--max-steps", type=int, default=0,
                       help="Maximum steps per episode (0 = no limit)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for evaluation")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                       help="Output directory for videos and logs")
    parser.add_argument("--save-video", action="store_true",
                       help="Save evaluation videos")
    
    # Model arguments
    parser.add_argument("--model-type", type=str, default="auto",
                       choices=["standard", "dueling", "atari", "auto"],
                       help="Model architecture type (auto-detect if 'auto')")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ENHANCED DQN EVALUATION")
    print("=" * 60)
    print(f"Model Path: {args.model_path}")
    print(f"Environment: {args.env_name}")
    print(f"Episodes: {args.episodes}")
    print(f"Save Videos: {args.save_video}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Random Seed: {args.seed}")
    
    # Detect environment type and run appropriate evaluation
    if "ALE/" in args.env_name or "Pong" in args.env_name or "Breakout" in args.env_name:
        print("Detected Atari environment")
        evaluate_atari(args)
    else:
        print("Detected CartPole environment")
        evaluate_cartpole(args)

if __name__ == "__main__":
    main()