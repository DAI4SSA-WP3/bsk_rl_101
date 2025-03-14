import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from torch.distributions import Categorical
warnings.simplefilter("ignore")

############################# Data Store ####################################################


class PPOMemory():
    """
    Memory for PPO
    """

    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.vals = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return (np.array(self.states), np.array(self.actions),
                np.array(self.action_probs), np.array(self.vals),
                np.array(self.rewards), np.array(self.dones), batches)

    def store_memory(self, state, actions, action_probs, val, reward, done):
        self.states.append(state)             # Store the entire combined state
        # Store the tuple of actions as a single entity
        self.actions.append(actions)
        # Store the tuple of action probabilities
        self.action_probs.append(action_probs)
        self.rewards.append(reward)
        self.vals.append(val)                 # Store the single value estimate
        self.dones.append(done)

    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.action_probs.clear()
        self.rewards.clear()
        self.vals.clear()
        self.dones.clear()


############################ Actor Network ######################################

class ActorNwk(nn.Module):
    def __init__(self, input_dim, action_dims, adam_lr, checkpoint_file, device, hidden1_dim=256, hidden2_dim=256):
        super(ActorNwk, self).__init__()

        self.actor_base = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU()
        )

        # Separate linear layers for each action dimension in multi-discrete actions
        self.action_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden2_dim, dim), nn.Softmax(dim=-1))
            for dim in action_dims
        ])

        self.checkpoint_file = checkpoint_file
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=adam_lr)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        base_out = self.actor_base(state)
        dists = [Categorical(head(base_out)) for head in self.action_heads]
        return dists

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


############################### Critic Network ######################################

class CriticNwk(nn.Module):
    def __init__(self, input_dim, adam_lr, checkpoint_file, device, hidden1_dim=256, hidden2_dim=256):
        super(CriticNwk, self).__init__()

        self.critic_nwk = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, 1)
        )

        self.checkpoint_file = checkpoint_file
        self.optimizer = torch.optim.Adam(
            params=self.critic_nwk.parameters(), lr=adam_lr)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        out = self.critic_nwk(state)
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

# Agent Class


class Agent():
    def __init__(self, gamma, policy_clip, lamda, adam_lr, n_epochs, batch_size, state_dim, action_dims, device, entropy_coef=0.1, diversity_coef=0.005):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.lamda = lamda
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.diversity_coef = diversity_coef

        self.actor = ActorNwk(input_dim=state_dim, action_dims=action_dims,
                              adam_lr=adam_lr, checkpoint_file='tmp/actor', device=device)
        self.critic = CriticNwk(
            input_dim=state_dim, adam_lr=adam_lr, checkpoint_file='tmp/critic', device=device)
        self.memory = PPOMemory(batch_size)

    def store_data(self, state, actions, action_probs, val, reward, done):
        self.memory.store_memory(
            state, actions, action_probs, val, reward, done)

    def save_models(self):
        print('... Saving Models ......')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... Loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.actor.device)
        dists = self.actor(state)

        actions = []
        action_probs = []
        for dist in dists:
            action = dist.sample()
            actions.append(action.item())
            action_probs.append(dist.log_prob(action).item())

        value = self.critic(state)
        value = torch.squeeze(value).item()

        return actions, action_probs, value

    # calculate_advantage function
    def calculate_advantage(self, reward_arr, value_arr, dones_arr):
        time_steps = len(reward_arr)
        advantage = np.zeros(time_steps, dtype=np.float32)

        for t in range(time_steps - 1):
            discount = 1
            running_advantage = 0
            for k in range(t, time_steps - 1):
                if dones_arr[k]:
                    running_advantage += reward_arr[k] - value_arr[k]
                else:
                    running_advantage += reward_arr[k] + \
                        (self.gamma * value_arr[k + 1]) - value_arr[k]
                running_advantage *= discount
                discount *= self.gamma * self.lamda

            advantage[t] = running_advantage
        advantage = torch.tensor(
            advantage, dtype=torch.float32).to(self.actor.device)
        return advantage

    # Revised learn function
    def learn(self):
        entropy_coef = self.entropy_coef
        diversity_coef = self.diversity_coef

        actor_loss_total = 0
        critic_loss_total = 0
        entropy_total = 0
        diversity_loss_total = 0
        total_loss_total = 0
        num_batches = 0

        for _ in range(self.n_epochs):
            # Retrieve data and generate advantage array
            state_arr, action_arr, old_prob_arr, value_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
            advantage_arr = self.calculate_advantage(
                reward_arr, value_arr, dones_arr)
            values = torch.tensor(
                value_arr, dtype=torch.float32).to(self.actor.device)

            for batch in batches:
                batch = batch.tolist()
                states = torch.tensor(
                    state_arr[batch], dtype=torch.float32).to(self.actor.device)
                old_probs = torch.tensor(
                    old_prob_arr[batch], dtype=torch.float32).to(self.actor.device)
                actions = torch.tensor(
                    action_arr[batch], dtype=torch.float32).to(self.actor.device)

                # Forward pass through the actor to get new action distributions
                dists = self.actor(states.view(states.size(0), -1))
                critic_value = self.critic(
                    states.view(states.size(0), -1)).squeeze()

                # Calculate log probabilities for each action in the multi-action setting
                log_probs = [dists[i].log_prob(
                    actions[:, i]) for i in range(actions.shape[1])]
                new_probs = torch.stack(log_probs, dim=1)
                prob_ratio = torch.exp(new_probs - old_probs)

                # Calculate actor loss with clipped objective
                weighted_probs = advantage_arr[batch].unsqueeze(1) * prob_ratio
                weighted_clipped_probs = torch.clamp(
                    prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage_arr[batch].unsqueeze(1)
                actor_loss = -torch.min(weighted_probs,
                                        weighted_clipped_probs).mean()

                # Compute critic loss
                returns = advantage_arr[batch] + values[batch]
                critic_loss = F.mse_loss(returns, critic_value)

                # Entropy regularization for encouraging exploration within each distribution
                entropies = [dist.entropy() for dist in dists]
                total_entropy = torch.stack(entropies, dim=1).mean()

                # Calculate diversity loss between distributions (pairwise KL divergence)
                diversity_loss = 0
                num_distributions = len(dists)
                for i in range(num_distributions):
                    for j in range(i + 1, num_distributions):
                        kl_div = torch.distributions.kl.kl_divergence(
                            dists[i], dists[j]).mean()
                        diversity_loss += kl_div
                # Average over pairs
                if num_distributions == 1:
                    diversity_loss = torch.tensor(
                        [0.0], dtype=torch.float32).to(self.actor.device)
                else:
                    diversity_loss /= (num_distributions *
                                       (num_distributions - 1) / 2)

                # Total loss with entropy and diversity encouragement
                total_loss = actor_loss + critic_loss - entropy_coef * \
                    total_entropy - diversity_coef * diversity_loss

                # Optimization step
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                # Accumulate losses for logging
                actor_loss_total += actor_loss.item()
                critic_loss_total += critic_loss.item()
                entropy_total += total_entropy.item()
                diversity_loss_total += diversity_loss.item()
                total_loss_total += total_loss.item()
                num_batches += 1

        self.memory.clear_memory()

        # Return average losses across all batches
        return {
            "actor_loss": actor_loss_total / num_batches,
            "critic_loss": critic_loss_total / num_batches,
            "entropy": entropy_total / num_batches,
            "diversity_loss": diversity_loss_total / num_batches,
            "total_loss": total_loss_total / num_batches,
        }
