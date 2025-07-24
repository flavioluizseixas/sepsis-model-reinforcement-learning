import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals


class DDDQNAgent:
    def __init__(self, state_dim, action_dim, replay_buffer, gamma=0.99, lr=1e-3, batch_size=64, tau=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer = replay_buffer
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def sample_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)

        return states, actions, rewards, next_states

    def train(self, episodes=100):
        for ep in range(episodes):
            if len(self.buffer) < self.batch_size:
                continue

            states, actions, rewards, next_states = self.sample_batch()

            with torch.no_grad():
                # DDDQN: usar a policy net para escolher a ação, mas target net para o valor
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
                target_q = rewards + self.gamma * next_q

            current_q = self.q_net(states).gather(1, actions)
            loss = self.loss_fn(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Atualização suave da rede alvo
            for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if ep % 10 == 0:
                print(f"🎯 Episódio {ep} | Loss: {loss.item():.4f}")

    def evaluate(self, env):
        self.q_net.eval()  # garante que dropout/batchnorm (se existirem) estão em modo avaliação
        transitions = env.get_buffer()
        total_q = 0.0
        count = 0

        for s, _, _, _ in transitions:
            state = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state)
                chosen_action = q_values.argmax().item()
                q_val = q_values[0, chosen_action].item()
                total_q += q_val
                count += 1

        avg_q = total_q / count if count > 0 else 0.0
        print(f"✅ Avaliação: valor médio Q das ações escolhidas: {avg_q:.4f}")

        match_count = 0
        for s, a, _, _ in transitions:
            state = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state)
                chosen_action = q_values.argmax().item()
                if chosen_action == a:
                    match_count += 1
        acc = match_count / len(transitions)
        print(f"🧠 Similaridade com histórico (acurácia supervisionada): {acc:.2%}")
