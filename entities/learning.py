
# Reinforcement Learning
class QLearning:
    def __init__(self, brain, learning_rate=0.01, discount_factor=0.99):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.memory = []
        self.brain = brain
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def get_target_q_value(self, reward, next_state):
        # Bellman Equation
        target_q_value = reward + self.discount_factor * torch.max(self.brain(next_state))
        return target_q_value

    def replay(self):
        for state, action, reward, next_state in self.memory:
            predicted_q_value = self.brain(inputs)
            target_q_value = target_q_value.unsqueeze(0).expand_as(predicted_q_value)
            loss = F.mse_loss(predicted_q_value, target_q_value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.memory.clear()

# Neural Networking
class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()
        self.fc1 = nn.Linear(3, 16)  # Input: relative x and y to the closest food
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)  # Output: movement direction in x and y

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)   

