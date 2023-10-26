import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import random

class EntityNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(EntityNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=0)
        
class Entity:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.energy = 100
        self.neural_network = EntityNN(input_size=4, output_size=5)
        self.optimizer = optim.Adam(self.neural_network.parameters(), lr=0.01)
        self.memory = []
        
    def get_state(self, food_sources):
        # Find distance to nearest food source, as an example
        nearest_food_dist = min([((self.x - fx)**2 + (self.y -fy)**2)**0.5 for fx, fy in food_sources])
        return torch.tensor([self.x, self.y, self.energy, nearest_food_dist], dtype=torch.float32)
        
    def decide_action(self, food_sources, epsilon=0.1):
        state = self.get_state(food_sources)
        return self.epsilon_greedy_action(state, epsilon)
    
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
        # Default negative reward for energy expenditure
        reward = -1

        # If the entity interacts
        if actions == 4:
            interaction_reward = 0

            # Check for food within interaction range
            for food in food_sources:
                if abs(self.x - food[0]) < 5 and abs(self.y - food[1]) < 5:
                    interaction_reward += 10 # Positive reward for finding food
                    self.energy += 10 # Increase energy for consuming food
                    food_sources.remove(food)
                    break
            
            # If no food was found within interaction range, slightly penalize
            if interaction_reward == 0:
                reward -= 2

        # If the entity moves, check if it's moving towards food when energy is low
        elif self.energy < 50:
            nearest_food_dist = min([((self.x - fx)**2 + (self.y - fy)**2)**0.5 for fx, fy in food_sources])
            new_x, new_y = self.x, self.y
        if action == 0:  # Move Up
            new_y -= 1
        elif action == 1:  # Move Down
            new_y += 1
        elif action == 2:  # Move Left
            new_x -= 1
        elif action == 3:  # Move Right
            new_x += 1
        new_nearest_food_dist = min([((new_x - fx)**2 + (new_y - fy)**2)**0.5 for fx, fy in food_sources])
        
        # If moving closer to food, slightly reward the entity
        if new_nearest_food_dist < nearest_food_dist:
            reward += 2
        else:
            reward -= 2  # Penalize if moving away from food

    # Deduct energy for every action
    self.energy -= 1

    return reward

    def epsilon_greedy_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 4) # 5 possible actions
        else:
            with torch.no_grad():
                q_values = self.neural_network(state)
            return q_values.argmax().item()
        
    def store_experiences(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        
    def train(self, batch_size=32, gamma=0.99):
        # Check if enough experiences are available
        if len(self.memory) < batch_size:
            return

        # Sample experiences from memory
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)

        # Compute predicted Q-values
        predicted_q_values = self.neural_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.neural_network(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(predicted_q_values, target_q_values)

        # Optimize neural network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def store_experience(self, state, action, reward):
        self.memory.append((state, action, reward))
    

# Intialize the environment
pygame.init()
screen = pygame.display.set_mode((800, 600))

# Create entities and food sources
entities = [Entity(random.randint(0, 800), random.randint(0, 600)) for _ in range(2)]
food_sources = [(random.randint(0, 800), random.randint(0, 600)) for _ in range(10)]

# Simulation loop
running = True
epsilon - 1.0 # For epsilon-greedy exploration
epsilon_decay = 0.995 # Decay rate for epsilon

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update entities
    for entity in entities:
        action = entity.decide_action(food_sources, epsilon)
        entity.perform_action(action, food_sources)
        reward = entity.receive_reward(action, food_sources)
        next_state = entity.get_state(food_sources)
        entity.store_experience(entity.get_state(food_sources), action, reward, next_state)
        entity.train()

    # Decay epsilon
    epsilon *= epsilon_decay

    # Render entities and food sources on the screen
     screen.fill((200, 200, 200))
    for entity in entities:
        pygame.draw.circle(screen, (0, 0, 0), (entity.x, entity.y), 5)
    for food in food_sources:
        pygame.draw.circle(screen, (0, 255, 0), food, 3)
    pygame.display.flip()

pygame.quit()

