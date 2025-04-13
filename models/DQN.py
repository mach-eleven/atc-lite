import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(state).to(device),
            torch.LongTensor(action).to(device),
            torch.FloatTensor(reward).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).to(device),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_shape,
        buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update_freq=1000,
    ):
        self.state_dim = state_dim
        self.action_shape = action_shape
        self.action_dim = np.prod(action_shape)

        self.q_net = QNetwork(state_dim, self.action_dim).to(device)
        self.target_q_net = QNetwork(state_dim, self.action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.update_counter = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            action_index = random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_net(state_tensor)
            action_index = q_values.argmax().item()

        return np.unravel_index(action_index, self.action_shape)

    def store_transition(self, state, action, reward, next_state, done):
        flat_action = np.ravel_multi_index(action, self.action_shape)
        self.buffer.push(state, flat_action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        q_values = self.q_net(states)
        next_q_values = self.target_q_net(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        max_next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + self.gamma * max_next_q_value * (1 - dones)

        loss = F.mse_loss(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_q_net.load_state_dict(torch.load(path))