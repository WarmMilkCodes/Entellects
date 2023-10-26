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

class TimeSystem:
    def __init__(self, initial_time=0, time_scale=24):
        self.current_time = initial_time # time in hours
        self.time_scale = time_scale # how many simulation hours pass per real-world hour

    def update(self, delta_time):
        self.current_time += delta_time / 3600 * self.time_scale

    def get_time_of_dau(self):
        hour = self.current_time % 24
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 24:
            return "evening"
        else:
            return "night"

    def get_season(self):
        day_of_year = (self.current_time // 24) % 365
        if 0 <= day_of_year < 91:
            return "spring"
        elif 91 <= day_of_year < 182:
            return "summer"
        elif 182 <= day_of_year < 273:
            return "fall"
        else:
            return "winter"
        
class Entity:
    def __init__(self, x, y, gender=None, name=None):
        self.x = x
        self.y = y
        self.energy = 100
        self.age = 0
        self.gender = gender or random.choice(['male', 'female'])
        self.neural_network = EntityNN(input_size=4, output_size=5)
        self.optimizer = optim.Adam(self.neural_network.parameters(), lr=0.01)
        self.memory = []
        self.name = name or self.generate_name()

    def check_mortality(self, environmental_factor=1.0):
        # Energy-based mortality
        if self.energy <= 0:
            return True

        # Envornmental factor mortality (eg, harsh winter)
        if self.energy < 10 and random.random() < 0.01 * environmental_factor:
            return True

        # Random chance mortality
        if random.random() < 0.001: # 0.01% chance every tick
            return True
        
        return False

    def gather_resources(self, resource):
        if self.nearby(resource):
            self.resource[resource.type] += resource.gather()

    def construct_shelter(self):
        if self.resources['wood'] >= 10: # example condition
            self.shelter = Shelter(quality=self.resources['wood'])
            self.resources['wood'] -= 10

    def seek_shelter(self):
        if self.shelter and self.shelter.quality > 0:
            self.in_shelter = True
        else:
            # Find nearest shelter or community
            pass
        

    def generate_name(self):
        syllables = ['ka', 'ri', 'to', 'na', 'lu', 'mi']
        name_length = random.randint(2, 3)
        return ''.join(random.choice(syllables) for _ in range(name_length))

    def can_reproduce_with(self, other_entity):
        # Basic criteria for reproduction
        if self.gender != other_entity.gender and self.age > 10 and other_entity.age > 10:
            return True
        return False

    def reproduce(self, other_entity):
        if self.can_reproduce_with(other_entity):
            # Determine offspring properties based on parents
            child_x = (self.x + other_entity.x) // 2
            child_y = (self.y + other_entity.y) // 2
            child = Entity(child_x, child_y)
            child_name = self.name[:2] + other_entity.name[-2:]
            child = Entity(child_x, child_y, name=child_name)
            return child
        
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
    
# Initialize the time system
time_system = TimeSystem()

# Initialize the environment
pygame.init()
screen = pygame.display.set_mode((800, 600))

# Load font for text display
font = pygame.font.SysFont(None, 36) # default font, size 36

# Create entities and food sources
entities = [Entity(random.randint(0, 800), random.randint(0, 600)) for _ in range(2)]
food_sources = [(random.randint(0, 800), random.randint(0, 600)) for _ in range(10)]

# Simulation loop
running = True
delta_time = 1.0
epsilon = 1.0 # For epsilon-greedy exploration
epsilon_decay = 0.995 # Decay rate for epsilon

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calculate delta_time (time passed since last frame)
    current_ticks = pygame.time.get_ticks()
    delta_time = (current_ticks - previous_ticks) / 1000.0 # seconds
    previous_ticks = current_ticks                  

    # Update time system
    time_system.update(delta_time * 60) # 1 minute real time = 1 hour in sim
    current_season = time_system.get_season()
    current_time_of_day = time_system.get_time_of_day()

    # Print current time and season (for testing)
    print(f"Current time: {time_system.current_time:.2f} hours, Season: {time_system.get_season()}")

    # Update entities
    for entity in entities:
        #Entity ages with the passage of time
        entity.age += delta_time / 3600.0 # Conver seconds to hours
        action = entity.decide_action(food_sources, epsilon)
        entity.perform_action(action, food_sources)
        reward = entity.receive_reward(action, food_sources)
        next_state = entity.get_state(food_sources)
        entity.store_experience(entity.get_state(food_sources), action, reward, next_state)
        entity.train()
        if entity.check_morality():
            entities.remove(entity)

    # Seasonal dynamics (ex. adjusting food regen rate)
    if current_season in ['spring', 'summer']:
        if random.random() < 0.05: # 5% chance for a new food source to appear
            food_sources.append((random.randit(0, 800), random.randit(0, 600)))
    elif current_season in ['fall', 'winter']:
        if random.random() < 0.02: # 2% chance for a new food source to appear
            food_sources.append((random.randint(0, 800), random.randint(0, 600)))
        
    # Check for entities in proximity and allow them to reproduce
    new_offsprings = []
    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities):
            if i != j and abs(entity1.x - entity2.x) < 5 and abs(entity1.y - entity2.y) < 5:
                child = entity1.reproduce(entity2)
                if child:
                    new_offsprings.append(child)
    entities.extend(new_offsprings)

    # Decay epsilon
    epsilon *= epsilon_decay

    # Render in-game date and time
    in_game_days = int(time_system.current_time // 24)
    in_game_hours = int(time_system.current_time % 24)
    time_text = f"Day {in_game_days}, {in_game_hours:02d}:00 {time_system.get_time_of_day().capitalize()}, {time_system.get_season().capitalize()}"
    time_surface = font.render(time_text, True, (0, 0, 0))
    screen.blit(time_surface, (10, 10))

    # Render entities and food sources on the screen
    screen.fill((200, 200, 200))
    for entity in entities:
        pygame.draw.circle(screen, (0, 0, 0), (entity.x, entity.y), 5)
    for food in food_sources:
        pygame.draw.circle(screen, (0, 255, 0), food, 3)
    pygame.display.flip()

pygame.quit()

