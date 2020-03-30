import random
from collections import namedtuple, deque
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork
from default_hyperparameters import SEED, BUFFER_SIZE, BATCH_SIZE, START_SINCE,\
                                    GAMMA, T_UPDATE, TAU, LR, WEIGHT_DECAY, UPDATE_EVERY,\
                                    A, INIT_BETA, P_EPS, N_STEPS, V_MIN, V_MAX,\
                                    CLIP, N_ATOMS, INIT_SIGMA, LINEAR, FACTORIZED

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=SEED, batch_size=BATCH_SIZE,
                 buffer_size=BUFFER_SIZE, start_since=START_SINCE, gamma=GAMMA, target_update_every=T_UPDATE,
                 tau=TAU, lr=LR, weight_decay=WEIGHT_DECAY, update_every=UPDATE_EVERY, priority_eps=P_EPS,
                 a=A, initial_beta=INIT_BETA, n_multisteps=N_STEPS,
                 v_min=V_MIN, v_max=V_MAX, clip=CLIP, n_atoms=N_ATOMS,
                 initial_sigma=INIT_SIGMA, linear_type=LINEAR, factorized=FACTORIZED, **kwds):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            batch_size (int): size of each sample batch
            buffer_size (int): size of the experience memory buffer
            start_since (int): number of steps to collect before start training
            gamma (float): discount factor
            target_update_every (int): how often to update the target network
            tau (float): target network soft-update parameter
            lr (float): learning rate
            weight_decay (float): weight decay for optimizer
            update_every (int): update(learning and target update) interval
            priority_eps (float): small base value for priorities
            a (float): priority exponent parameter
            initial_beta (float): initial importance-sampling weight
            n_multisteps (int): number of steps to consider for each experience
            v_min (float): minimum reward support value
            v_max (float): maximum reward support value
            clip (float): gradient norm clipping (`None` to disable)
            n_atoms (int): number of atoms in the discrete support distribution
            initial_sigma (float): initial noise parameter weights
            linear_type (str): one of ('linear', 'noisy'); type of linear layer to use
            factorized (bool): whether to use factorized gaussian noise in noisy layers
        """
        if kwds != {}:
            print("Ignored keyword arguments: ", end='')
            print(*kwds, sep=', ')
        assert isinstance(state_size, int)
        assert isinstance(action_size, int)
        assert isinstance(seed, int)
        assert isinstance(batch_size, int) and batch_size > 0
        assert isinstance(buffer_size, int) and buffer_size >= batch_size
        assert isinstance(start_since, int) and batch_size <= start_since <= buffer_size
        assert isinstance(gamma, (int, float)) and 0 <= gamma <= 1
        assert isinstance(target_update_every, int) and target_update_every > 0
        assert isinstance(tau, (int, float)) and 0 <= tau <= 1
        assert isinstance(lr, (int, float)) and lr >= 0
        assert isinstance(weight_decay, (int, float)) and weight_decay >= 0
        assert isinstance(update_every, int) and update_every > 0
        assert isinstance(priority_eps, (int, float)) and priority_eps >= 0
        assert isinstance(a, (int, float)) and 0 <= a <= 1
        assert isinstance(initial_beta, (int, float)) and 0 <= initial_beta <= 1
        assert isinstance(n_multisteps, int) and n_multisteps > 0
        assert isinstance(v_min, (int, float)) and isinstance(v_max, (int, float)) and v_min < v_max
        if clip: assert isinstance(clip, (int, float)) and clip >= 0
        assert isinstance(n_atoms, int) and n_atoms > 0
        assert isinstance(initial_sigma, (int, float)) and initial_sigma >= 0
        assert isinstance(linear_type, str) and linear_type.strip().lower() in ('linear', 'noisy')
        assert isinstance(factorized, bool)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.state_size          = state_size
        self.action_size         = action_size
        self.seed                = seed
        self.batch_size          = batch_size
        self.buffer_size         = buffer_size
        self.start_since         = start_since
        self.gamma               = gamma
        self.target_update_every = target_update_every
        self.tau                 = tau
        self.lr                  = lr
        self.weight_decay        = weight_decay
        self.update_every        = update_every
        self.priority_eps        = priority_eps
        self.a                   = a
        self.beta                = initial_beta
        self.n_multisteps        = n_multisteps
        self.v_min               = v_min
        self.v_max               = v_max
        self.clip                = clip
        self.n_atoms             = n_atoms
        self.initial_sigma       = initial_sigma
        self.linear_type         = linear_type.strip().lower()
        self.factorized          = factorized

        # Distribution
        self.supports = torch.linspace(v_min, v_max, n_atoms, device=device)
        self.delta_z  = (v_max - v_min) / (n_atoms - 1)

        # Q-Network
        self.qnetwork_local  = QNetwork(state_size, action_size, n_atoms, linear_type, initial_sigma, factorized).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, n_atoms, linear_type, initial_sigma, factorized).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr, weight_decay=weight_decay)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, n_multisteps, gamma, a)
        # Initialize time step (for updating every UPDATE_EVERY steps and TARGET_UPDATE_EVERY steps)
        self.u_step = 0
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.u_step = (self.u_step + 1) % self.update_every
        if self.u_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= self.start_since:
                experiences, target_discount, is_weights, indices = self.memory.sample(self.beta)
                new_priorities = self.learn(experiences, is_weights, target_discount)
                self.memory.update_priorities(indices, new_priorities)

        # update the target network every TARGET_UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.target_update_every
        if self.t_step == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            z_probs       = F.softmax(self.qnetwork_local(state), dim=-1)
            action_values = self.supports.mul(z_probs).sum(dim=-1, keepdim=False)

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        return random.choice(np.arange(self.action_size))

    def learn(self, experiences, is_weights, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            is_weights (torch.Tensor): tensor of importance-sampling weights
            gamma (float): discount factor for the target max-Q value

        Returns
        =======
            new_priorities (List[float]): list of new priority values for the given sample
        """
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            rows         = tuple(range(next_states.size(0)))
            a_argmax     = F.softmax(self.qnetwork_local(next_states), dim=2)\
                               .mul(self.supports)\
                               .sum(dim=2, keepdim=False)\
                               .argmax(dim=1, keepdim=False)
            p            = F.softmax(self.qnetwork_target(next_states)[rows, a_argmax], dim=1)
            tz_projected = torch.clamp(rewards + (1 - dones) * gamma * self.supports, min=self.v_min, max=self.v_max)
            # """
            b            = (tz_projected - self.v_min) / self.delta_z
            u            = b.ceil()
            l            = b.floor()
            u_updates    = b - l + u.eq(l).type(u.dtype) # fixes the problem when having b == u == l
            l_updates    = u - b
            indices_flat = torch.cat((u.long(), l.long()), dim=1)
            indices_flat = indices_flat.add(
                               torch.arange(start=0,
                                            end=b.size(0) * b.size(1),
                                            step=b.size(1),
                                            dtype=indices_flat.dtype,
                                            layout=indices_flat.layout,
                                            device=indices_flat.device).unsqueeze(1)
                           ).view(-1)
            updates_flat = torch.cat((u_updates.mul(p), l_updates.mul(p)), dim=1).view(-1)
            target_distributions = torch.zeros_like(p)
            target_distributions.view(-1).index_add_(0, indices_flat, updates_flat)
            """
            b = ((tz_projected - V_MIN) / self.delta_z).t() # transpose for later for-loop convenience
            u = b.ceil()
            l = b.floor()
            u_updates = b - l + u.eq(l).type(u.dtype)
            l_updates = u - b
            target_distributions = torch.zeros_like(p)
            for u_indices, l_indices, u_update, l_update, prob in zip(u.long(), l.long(), u_updates, l_updates, p.t()):
                target_distributions[rows, u_indices] += u_update * prob
                target_distributions[rows, l_indices] += l_update * prob
            """

        pred_distributions = self.qnetwork_local(states)
        pred_distributions = pred_distributions.gather(dim=1, index=actions.unsqueeze(1).expand(-1, -1, pred_distributions.size(2))).squeeze(1)

        """
        cross_entropy = target_distributions.mul(pred_distributions.exp().sum(dim=-1, keepdim=True).log() - pred_distributions).sum(dim=-1, keepdim=False)
        new_priorities = cross_entropy.detach().add(self.priority_eps).cpu().numpy()
        loss = cross_entropy.mul(is_weights.view(-1)).mean()
        """
        kl_divergence = F.kl_div(F.log_softmax(pred_distributions, dim=-1), target_distributions, reduce=False).sum(dim=-1, keepdim=False)
        new_priorities = kl_divergence.detach().add(self.priority_eps).cpu().numpy()
        loss = kl_divergence.mul(is_weights.view(-1)).mean()
#         """

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.clip)
        self.optimizer.step()

        return new_priorities

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, n_multisteps, gamma, a):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            n_multisteps (int): number of time steps to consider for each experience
            gamma (float): discount factor
            a (float): priority exponent parameter
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_multisteps = n_multisteps
        self.gamma = gamma
        self.a = a
        self.memory = deque(maxlen=buffer_size)
        self.priorities_a = deque(maxlen=buffer_size)
        self.multistep_collector = deque(maxlen=n_multisteps)
        self.max_priority_a = 1.
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.multistep_collector.append(e)
        if len(self.multistep_collector) == self.n_multisteps:
            self.memory.append(tuple(self.multistep_collector))
            self.priorities_a.append(self.max_priority_a)
        if done:
            self.multistep_collector.clear()

    def sample(self, beta):
        """Randomly sample a batch of experiences from memory.

        Params
        ======
            beta (int or float): parameter used for calculating importance-priority weights

        Returns
        =======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            target_discount (float): discount factor for target max-Q value
            is_weights (torch.Tensor): tensor of importance-sampling weights
            indices (np.ndarray): sample indices"""
        probs = np.divide(self.priorities_a, sum(self.priorities_a))

        indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False, p=probs)

        discounts = np.power(self.gamma, np.arange(self.n_multisteps + 1))
        target_discount = float(discounts[-1])

        experiences = tuple(zip(*[self.memory[i] for i in indices if self.memory[i] is not None]))


        first_states = torch.from_numpy(np.array([e[0] for e in experiences[0]])).float().to(device)
        actions = torch.from_numpy(np.array([e[1] for e in experiences[0]]).reshape((-1, 1))).long().to(device)
        rewards = torch.from_numpy(
                      np.sum(
                          np.multiply(
                              np.array([[e[2] for e in experiences_step] for experiences_step in experiences]).transpose(), discounts[:-1]
                          ), axis=1, keepdims=True
                      )
                  ).float().to(device)
        last_states = torch.from_numpy(np.array([e[3] for e in experiences[-1]])).float().to(device)
        dones = torch.from_numpy(np.array([e[4] for e in experiences[-1]], dtype=np.uint8).reshape((-1, 1))).float().to(device)

        is_weights = [probs[i] for i in indices if self.memory[i] is not None]
        is_weights = np.power(np.multiply(is_weights, len(self.memory)), -beta)
        is_weights = torch.from_numpy(np.divide(is_weights, max(is_weights)).reshape((-1, 1))).float().to(device)

        return (first_states, actions, rewards, last_states, dones), target_discount, is_weights, indices

    def update_priorities(self, indices, new_priorities):
        """Update the priorities for the experiences of given indices to the given new values.

        Params
        ======
            indices (array_like): indices of experience priorities to update
            new_priorities (array-like): new priority values for given indices"""
        new_priorities_a = np.power(new_priorities, self.a)
        for i, new_priority_a in zip(indices, new_priorities_a):
            self.priorities_a[i] = float(new_priority_a)
        self.max_priority_a = max(self.priorities_a)

    def reset_multisteps(self):
        self.multistep_collector.clear()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
