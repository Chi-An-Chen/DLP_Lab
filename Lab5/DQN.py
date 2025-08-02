"""
Author: Chi-An Chen
Date: 2025-08-02
Description: Training DQN model
"""
import os
import cv2
import time
import math
import wandb
import torch
import random
import ale_py
import argparse
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
from tqdm import tqdm, trange

gym.register_envs(ale_py)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

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

class AtariPreprocessor:
    """
    Preprocessing the state input of DQN for Atari
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        """Convert RGB observation to grayscale and resize"""
        if len(obs.shape) == 3:  # RGB image
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:  # Already grayscale
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0  # Normalize to [0, 1]

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

class CartpolePreprocessor:
    """Enhanced preprocessor with state normalization"""
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
        state = np.array(state, dtype=np.float32)
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

class PrioritizedReplayBuffer:
    """
    Enhanced Prioritized Experience Replay with improved sampling
    Reference: Schaul et al., 2016 - https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def add(self, transition, error=None):
        """Add new transition to buffer with priority"""
        priority = self.max_priority if error is None else (abs(error) + 1e-6) ** self.alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        
        self.priorities[self.pos] = priority
        self.max_priority = max(self.max_priority, priority)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """Sample batch based on priorities"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
        
        # Calculate sampling probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

class NoiseScheduler:
    """Noise scheduler for better exploration"""
    def __init__(self, initial_epsilon=1.0, final_epsilon=0.01, decay_steps=50000):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = decay_steps
        self.step_count = 0
    
    def get_epsilon(self):
        """Get current epsilon using exponential decay"""
        if self.step_count >= self.decay_steps:
            return self.final_epsilon
        
        # Exponential decay
        decay_ratio = self.step_count / self.decay_steps
        epsilon = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * math.exp(-decay_ratio * 5)
        return epsilon
    
    def step(self):
        self.step_count += 1

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.env_name = env_name
        
        # Choose preprocessor based on environment
        if "CartPole" in env_name:
            self.preprocessor = CartpolePreprocessor()
            self.input_dim = 4
        else:  # Atari games
            self.preprocessor = AtariPreprocessor()
            self.input_dim = 4  # Frame stack

        # Device setup with better fallback
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Device: {self.device}")

        # Choose network architecture based on environment
        if "CartPole" in env_name:
            if args.use_dueling:
                self.q_net = DuelingDQN(input_dim=self.input_dim, num_actions=self.num_actions, 
                                    hidden_dim=args.hidden_dim, dropout_rate=args.dropout_rate).to(self.device)
                self.target_net = DuelingDQN(input_dim=self.input_dim, num_actions=self.num_actions,
                                        hidden_dim=args.hidden_dim, dropout_rate=args.dropout_rate).to(self.device)
                print("Using Dueling DQN for CartPole")
            else:
                self.q_net = DQN(input_dim=self.input_dim, num_actions=self.num_actions,
                            hidden_dim=args.hidden_dim, dropout_rate=args.dropout_rate).to(self.device)
                self.target_net = DQN(input_dim=self.input_dim, num_actions=self.num_actions,
                                    hidden_dim=args.hidden_dim, dropout_rate=args.dropout_rate).to(self.device)
                print("Using standard DQN for CartPole")
        else:  # Atari games
            if args.use_dueling:
                self.q_net = AtariDuelingCNNDQN(num_actions=self.num_actions, 
                                    dropout_rate=args.dropout_rate).to(self.device)
                self.target_net = AtariDuelingCNNDQN(num_actions=self.num_actions,
                                        dropout_rate=args.dropout_rate).to(self.device)
                print("Using Dueling CNN DQN for Atari")
            else:
                self.q_net = AtariCNNDQN(num_actions=self.num_actions,
                            dropout_rate=args.dropout_rate).to(self.device)
                self.target_net = AtariCNNDQN(num_actions=self.num_actions,
                                    dropout_rate=args.dropout_rate).to(self.device)
                print("Using standard CNN DQN for Atari")

        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Use different optimizers
        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.q_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            self.optimizer = optim.AdamW(self.q_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler(args)
        self.total_train_steps = 0

        # Use prioritized replay buffer
        if args.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(args.memory_size, alpha=args.alpha, beta=args.beta)
            print("Using prioritized experience replay")
        else:
            self.memory = deque(maxlen=args.memory_size)
            print("Using standard replay buffer")

        self.use_prioritized_replay = args.use_prioritized_replay
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        
        # Enhanced epsilon scheduling
        self.noise_scheduler = NoiseScheduler(
            initial_epsilon=args.epsilon_start,
            final_epsilon=args.epsilon_min,
            decay_steps=args.epsilon_decay_steps
        )

        self.env_count = 0
        self.train_count = 0
        self.best_reward = float('-inf')
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        self.use_double_dqn = args.use_double_dqn
        
        # Performance tracking
        self.recent_rewards = deque(maxlen=100)
        self.recent_eval_rewards = deque(maxlen=10)
        self.recent_losses = deque(maxlen=100)
        
        os.makedirs(self.save_dir, exist_ok=True)

    def _create_scheduler(self, args):
        """Create improved learning rate scheduler"""
        estimated_updates_per_episode = args.max_episode_steps * args.train_per_step
        total_train_steps = args.episodes * estimated_updates_per_episode
        warmup_steps = int(0.1 * total_train_steps)
        
        def get_lr_multiplier(current_step):
            if current_step < warmup_steps:
                # Warmup: linear growth from 0 to 1
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay: from 1 to min_lr_ratio
                progress = (current_step - warmup_steps) / float(max(1, total_train_steps - warmup_steps))
                min_lr_ratio = 0.1
                return max(min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, get_lr_multiplier)

    def select_action(self, state):
        """Enhanced action selection with noise scheduling"""
        epsilon = self.noise_scheduler.get_epsilon()
        
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Handle different state shapes for CartPole vs Atari
        if "CartPole" in self.env_name:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        else:  # Atari
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        self.q_net.train()
        return q_values.argmax().item()

    def run(self, episodes=1000):
        """Enhanced training loop with progress bar"""
        print(f"Start training with {episodes} episodes...")
        
        episode_pbar = trange(episodes, desc="Training", ncols=150)
        
        for ep in episode_pbar:
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0
            episode_loss = []

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state = self.preprocessor.step(next_obs)
                
                # Store transition
                if self.use_prioritized_replay:
                    self.memory.add((state, action, reward, next_state, done))
                else:
                    self.memory.append((state, action, reward, next_state, done))

                # Training
                for _ in range(self.train_per_step):
                    loss = self.train()
                    if loss is not None:
                        episode_loss.append(loss)
                        self.recent_losses.append(loss)

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1
                self.noise_scheduler.step()

            # Episode completed
            self.recent_rewards.append(total_reward)
            avg_recent_reward = np.mean(self.recent_rewards)
            avg_recent_loss = np.mean(self.recent_losses) if self.recent_losses else 0
            current_epsilon = self.noise_scheduler.get_epsilon()
            
            postfix_dict = {
                'Reward': f'{total_reward:.1f}',
                'Avg100': f'{avg_recent_reward:.1f}',
                'Best': f'{self.best_reward:.1f}',
                'Loss': f'{avg_recent_loss:.4f}',
                'ε': f'{current_epsilon:.3f}',
                'Steps': step_count,
                'Updates': self.train_count
            }
            episode_pbar.set_postfix(postfix_dict)
            
            # WandB logging
            if wandb.run:
                wandb.log({
                    "Episode": ep,
                    "Total Reward": total_reward,
                    "Average Recent Reward": avg_recent_reward,
                    "Episode Length": step_count,
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Epsilon": current_epsilon,
                    "Average Episode Loss": avg_recent_loss,
                    "Memory Size": len(self.memory.buffer) if self.use_prioritized_replay else len(self.memory),
                    "Learning Rate": self.optimizer.param_groups[0]['lr']
                })

            # Model saving
            if ep % 100 == 0 and ep > 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save({
                    'q_net_state_dict': self.q_net.state_dict(),
                    'target_net_state_dict': self.target_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'episode': ep,
                    'best_reward': self.best_reward
                }, model_path)
                episode_pbar.write(f"Saved model checkpoint to {model_path}")

            # Evaluation
            if ep % 20 == 0:
                eval_reward = self.evaluate()
                self.recent_eval_rewards.append(eval_reward)
                avg_eval_reward = np.mean(self.recent_eval_rewards)
                
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save({
                        'q_net_state_dict': self.q_net.state_dict(),
                        'target_net_state_dict': self.target_net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'episode': ep,
                        'best_reward': self.best_reward
                    }, model_path)
                    episode_pbar.write(f"New best model saved! Reward: {eval_reward:.2f}")
                
                if wandb.run:
                    wandb.log({
                        "Eval Reward": eval_reward,
                        "Average Eval Reward": avg_eval_reward,
                        "Best Reward": self.best_reward
                    })

        episode_pbar.close()
        print("Training Complete!")

    def evaluate(self, num_episodes=5):
        """Enhanced evaluation with multiple episodes"""
        total_rewards = []
        
        for _ in range(num_episodes):
            obs, _ = self.test_env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0

            while not done:
                # Handle different state shapes
                if "CartPole" in self.env_name:
                    state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                else:  # Atari
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                
                self.q_net.eval()
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()
                self.q_net.train()
                
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = self.preprocessor.step(next_obs)

            total_rewards.append(total_reward)

        return np.mean(total_rewards)

    def train(self):
        """Enhanced training with prioritized replay and double DQN"""
        memory_size = len(self.memory.buffer) if self.use_prioritized_replay else len(self.memory)
        if memory_size < self.replay_start_size:
            return None
        
        self.train_count += 1
        self.total_train_steps += 1
        
        # Sample from replay buffer
        if self.use_prioritized_replay:
            batch, indices, weights = self.memory.sample(self.batch_size)
            weights = torch.from_numpy(weights).to(self.device)
        else:
            batch = random.sample(self.memory, self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
        
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors - handle different shapes for CartPole vs Atari
        if "CartPole" in self.env_name:
            states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
            next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        else:  # Atari
            states = torch.from_numpy(np.stack(states).astype(np.float32)).to(self.device)
            next_states = torch.from_numpy(np.stack(next_states).astype(np.float32)).to(self.device)
        
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Current Q values
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate target Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use main network to select action, target network to evaluate
                next_actions = self.q_net(next_states).argmax(1)
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_net(next_states).max(1)[0]
            
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Calculate loss
        td_errors = target_q_values - current_q_values
        loss = (weights * td_errors.pow(2)).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update priorities if using prioritized replay
        if self.use_prioritized_replay:
            priorities = td_errors.abs().detach().cpu().numpy()
            self.memory.update_priorities(indices, priorities)

        # Update target network
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Log training metrics
        if wandb.run and self.train_count % 1000 == 0:
            wandb.log({
                "Training Loss": loss.item(),
                "Average Q Value": current_q_values.mean().item(),
                "Q Value Std": current_q_values.std().item(),
                "Target Q Average": target_q_values.mean().item(),
                "TD Error Mean": td_errors.abs().mean().item(),
                "Learning Rate": self.optimizer.param_groups[0]['lr']
            })

        return loss.item()


class EnvRunner:
    def __init__(self, env_name, wandb_apiKey, args):
        self.api_key = wandb_apiKey
        self.env_name = env_name
        self.args = args

    def run(self):
        run_name = f"enhanced-dqn-{self.env_name}-{int(time.time())}"
        print(f"Training in {self.env_name}, with run name: {run_name}")

        # Enhanced wandb configuration
        try:
            if self.api_key:
                wandb.login(key=self.api_key)
            wandb.init(
                project="Enhanced-DQN",
                name=run_name,
                save_code=True,
                config=vars(self.args)
            )
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            print("Continuing training without wandb logging...")

        agent = DQNAgent(env_name=self.env_name, args=self.args)
        agent.run(episodes=self.args.episodes)

        if wandb.run:
            wandb.finish()
        print(f"Training Complete!")


def create_env_and_get_specs(env_name):
    """Create environment and get its specifications"""
    env = gym.make(env_name)
    
    if hasattr(env.action_space, 'n'):
        num_actions = env.action_space.n
    else:
        raise ValueError("Only discrete action spaces are supported")
    
    # Get observation space info
    obs_space = env.observation_space
    if len(obs_space.shape) == 1:  # CartPole-like
        input_dim = obs_space.shape[0]
        env_type = "vector"
    elif len(obs_space.shape) == 3:  # Atari-like
        input_dim = obs_space.shape
        env_type = "image"
    else:
        raise ValueError(f"Unsupported observation space shape: {obs_space.shape}")
    
    env.close()
    return num_actions, input_dim, env_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced DQN Training")
    
    # Environment parameters
    parser.add_argument("--env", type=str, default="CartPole-v1", 
                       help="Environment name (e.g., CartPole-v1, ALE/Breakout-v5)")
    
    # Basic parameters
    parser.add_argument("--save-dir", type=str, default="./enhanced_results")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--memory-size", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--max-episode-steps", type=int, default=18000)
    parser.add_argument("--train-per-step", type=int, default=4)
    
    # Enhanced parameters
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--dropout-rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--optimizer", type=str, default="adamw", 
                       choices=["adam", "rmsprop", "adamw"], help="Optimizer type")
    
    # Exploration parameters
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay-steps", type=int, default=100000)
    
    # DQN variants
    parser.add_argument("--use-dueling", action="store_true", help="Use Dueling DQN")
    parser.add_argument("--use-double-dqn", action="store_true", help="Use Double DQN")
    parser.add_argument("--use-prioritized-replay", action="store_true", 
                       help="Use Prioritized Experience Replay")
    
    # Prioritized replay parameters
    parser.add_argument("--alpha", type=float, default=0.7, help="Prioritization exponent")
    parser.add_argument("--beta", type=float, default=0.5, help="Importance sampling exponent")
    
    # Training parameters
    parser.add_argument("--target-update-frequency", type=int, default=8000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    
    # Wandb parameters
    parser.add_argument("--wandb-key", type=str, default=None, 
                       help="Wandb API key (optional)")
    parser.add_argument("--no-wandb", action="store_true", 
                       help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Validate environment
    try:
        num_actions, input_dim, env_type = create_env_and_get_specs(args.env)
        print(f"Environment: {args.env}")
        print(f"Action space: {num_actions}")
        print(f"Observation space: {input_dim}")
        print(f"Environment type: {env_type}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    
    # Adjust parameters for Atari games
    if env_type == "image":
        args.max_episode_steps = 10000  # Atari games typically need more steps
        args.epsilon_decay_steps = 100000  # Longer exploration for Atari
        args.replay_start_size = 10000  # Larger replay start size
        args.memory_size = 1000000  # Larger memory for Atari
        print("Adjusted parameters for Atari environment")
    
    print("=" * 60)
    print("Training Configuration:")
    print(f"Environment: {args.env}")
    print(f"Model: {'Dueling' if args.use_dueling else 'Standard'} DQN")
    print(f"Double DQN: {args.use_double_dqn}")
    print(f"Prioritized Replay: {args.use_prioritized_replay}")
    print(f"Hidden Dimension: {args.hidden_dim}")
    print(f"Optimizer: {args.optimizer.upper()}")
    print(f"Learning Rate: {args.lr}")
    print(f"Episodes: {args.episodes}")
    print(f"Max Episode Steps: {args.max_episode_steps}")
    print("=" * 60)

    # Set wandb key if not provided via argument
    wandb_key = args.wandb_key or 'be5b6d66a5c7a55d302b8ae8df4be17fb2648bde'
    
    if args.no_wandb:
        wandb_key = None
        print("Wandb logging disabled")

    runner = EnvRunner(env_name=args.env, wandb_apiKey=wandb_key, args=args)
    runner.run()

"""
Usage examples:

# Train CartPole with all enhancements
python enhanced_dqn.py --env CartPole-v1 --use-dueling --use-double-dqn --use-prioritized-replay --lr 0.0005

# Train Atari Breakout
python enhanced_dqn.py --env ALE/Breakout-v5 --use-dueling --use-double-dqn --episodes 2000

# Train without wandb
python enhanced_dqn.py --env CartPole-v1 --no-wandb

# Custom configuration
python enhanced_dqn.py --env CartPole-v1 --hidden-dim 512 --lr 0.0001 --episodes 2000
"""