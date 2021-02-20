""""""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from memories.memories import ReplayMemory


class DQNAgent:
	def __init__(self, memory, batch_size, epsilon, epsilon_dec, epsilon_min, n_actions,
				 policy_net, target_net, gamma, lr):

		self.batch_size = batch_size
		self.epsilon = epsilon
		self.epsilon_dec = epsilon_dec
		self.epsilon_min = epsilon_min
		self.n_action_vec = [x for x in range(n_actions)]
		self.n_actions = n_actions
		self.policy_net = policy_net
		self.target_net = target_net
		self.gamma = gamma

		self.policy_net.to('cuda:0')
		self.target_net.to('cuda:0')

		self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
		self.loss_fn = nn.MSELoss()
		self.memory = memory

	def update_target_net(self):
		""""""
		self.target_net.load_state_dict(self.policy_net.state_dict())

	def store_memory(self, s, a, r, s_, t):
		self.memory.store(s, a, r, s_, t)

	def choose_action(self, s):
		if np.random.rand() < self.epsilon:
			return np.random.choice(self.n_action_vec)
		else:
			return torch.argmax(self.policy_net(torch.tensor(s).unsqueeze(0).float().to('cuda:0')))

	def learn(self):
		if len(self.memory.action) < self.batch_size:
			return

		s, a, r, s_, t = self.memory.sample(self.batch_size)

		q_sa = self.policy_net(torch.tensor(s).float().to('cuda:0'))
		q_sa_ = self.target_net(torch.tensor(s_).float().to('cuda:0'))

		# Zeroing out the qvalues of teminal states
		for i in range(t.shape[0]):
			if t[i] == 1:
				q_sa_[i, :] = torch.tensor([0. for _ in range(self.n_actions)])

		max_a, max_a_idxes = torch.max(q_sa_, dim=1)
		q_target = torch.tensor(r).to('cuda:0').float() + self.gamma * max_a
		loss = self.loss_fn(q_target, q_sa[[x for x in range(self.batch_size)], max_a_idxes])
		loss.backward()

		self.optimizer.step()

		if self.epsilon < self.epsilon_min:
			self.epsilon = self.epsilon_min
		else:
			self.epsilon -= self.epsilon_dec





