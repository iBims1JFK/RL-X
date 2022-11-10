import numpy as np


class ReplayBuffer():
    def __init__(self, capacity, nr_envs, os_shape, as_shape):
        self.os_shape = os_shape
        self.as_shape = as_shape
        self.capacity = capacity // nr_envs
        self.nr_envs = nr_envs
        self.states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.actions = np.zeros((self.capacity, nr_envs) + as_shape, dtype=np.float32)
        self.rewards = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.dones = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.pos = 0
        self.size = 0
    

    def add(self, states, next_states, actions, rewards, dones):
        self.states[self.pos] = states
        self.next_states[self.pos] = next_states
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    

    def sample(self, nr_samples, nr_batches):
        idx1 = np.random.randint(self.size, size=nr_samples * nr_batches)
        idx2 = np.random.randint(self.nr_envs, size=nr_samples * nr_batches)
        states = self.states[idx1, idx2].reshape((nr_batches, nr_samples) + self.os_shape)
        next_states = self.next_states[idx1, idx2].reshape((nr_batches, nr_samples) + self.os_shape)
        actions = self.actions[idx1, idx2].reshape((nr_batches, nr_samples) + self.as_shape)
        rewards = self.rewards[idx1, idx2].reshape((nr_batches, nr_samples))
        dones = self.dones[idx1, idx2].reshape((nr_batches, nr_samples))
        return states, next_states, actions, rewards, dones
