import torch
import torch.nn as nn
import torch.optim as optim

class EntityNN(nn.module):
    def __init__(self, input_size, output_size):
        super(EntityNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fg3(x), dim=0)
        
class Entity:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.energy = 100
        self.neural_network = EntityNN(input_size=3, output_size=5) # for simplicity
        self.optimizer = optim.Adam(self.neural_network.parameters(), lr=0.01)
        self.memory = []
        
    def get_state(self, food_sources):
        # Find distance to nearest food source, as an example
        nearest_food_dist = min([((self.x - fx)**2 + (self.y -fy)**2)**0.5 for fx, fy in food_sources])
        return torch.tensor([self.x, self.y, self.energy, nearest_food_dist], dtype=torch.float32)
        
    def decide_action(self):
        state = torch.tensor([self.x, self.y, self.energy], dtype=torch.float32)
        action_probs = self.neural_network(state)
        action = torch.multinomial(action_probs, 1).item()
        return action
    
    def perform_action(self, action, food_sources):
        # Placeholder logic for moving and interacting
        if action == 0: # move up
            self.y -= 1
        elif action == 1: # move down
            self.y += 1
        elif action == 2: # move left
            self.x -= 1
        elif action == 3: # move right
            self.x += 1
        elif action == 4: # interact
            pass # interaction logic (with food or other entities)
        
    def receive_reward(self, action, food_sources):
        # Placeholder logic for giving rewards based on actions and output
        reward = -1 # default negative reward for energy expenditure
        if action == 4: # interact
            for food in food_sources:
                if abs(self.x - food[0]) < 5 and abs(self.y - food[1]) < 5:
                    reward += 10 # postitive reward for finding food
                    food_sources.remove(food)
                    break
        return reward
        
    def store_experiences(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        
    def train(self, batch_size=32):
        # Placeholder for training logic using RL
        pass
    
    def store_experience(self, state, action, reward):
        self.memory.append((state, action, reward))
        
    def train(self):
        # Placeholder for training logic using RL
        pass
